"""
Bai3_v5_llm.py
Phương pháp 5: Phân loại cảm xúc IMDB bằng Mô hình ngôn ngữ lớn (LLM) miễn phí qua API
- Đánh giá trên tập con phân tầng 200 mẫu (seed=42) dùng chung với v1 -> v4
- So sánh Zero-shot prompting và Few-shot prompting
- Báo cáo đầy đủ: Accuracy, Macro-F1, Tỷ lệ đầu ra không hợp lệ, Độ trễ trung bình (s/mẫu)
- Hỗ trợ 3 API miễn phí:
    1) Groq   (khuyên dùng - nhanh nhất): tạo key tại https://console.groq.com/keys
       $env:GROQ_API_KEY="gsk_..."
       python Bai3_v5_llm.py --provider groq --mode both
    2) Gemini: tạo key tại https://aistudio.google.com/apikey
       $env:GEMINI_API_KEY="AIza..."
       python Bai3_v5_llm.py --provider gemini --mode both
    3) Hugging Face: tạo token tại https://huggingface.co/settings/tokens
       $env:HF_API_KEY="hf_..."
       python Bai3_v5_llm.py --provider hf --mode both
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from imdb_utils import (
    evaluate_predictions,
    get_stratified_subset,
    load_imdb_data,
)

SEED = 42
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

PROVIDER_DEFAULTS = {
    "groq": {
        "env_var": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant",
        "delay": 2.2,  # Free tier Groq ~30 requests/phút -> 2.2s an toàn không bị 429
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "delay": 4.5,  # Free tier Gemini ~10-15 RPM
    },
    "hf": {
        "env_var": "HF_API_KEY",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "delay": 1.5,
    },
}

ZERO_SHOT_TEMPLATE = """Classify the sentiment of the following movie review as either "positive" or "negative".
Return ONLY a valid JSON object with a single key "label" whose value is strictly "positive" or "negative". Do not include any explanation or extra text.

Review:
\"\"\"{review}\"\"\"

Output:"""

FEW_SHOT_TEMPLATE = """Classify the sentiment of a movie review as either "positive" or "negative".
Return ONLY a valid JSON object with a single key "label" whose value is strictly "positive" or "negative". Do not include any explanation.

Examples:
Review: "An engaging plot with brilliant performances from the entire cast. Easily one of the best films of the year."
Output: {{"label": "positive"}}

Review: "A complete waste of two hours. The dialogue was cringe-worthy and the ending made no sense."
Output: {{"label": "negative"}}

Review: "The visuals looked impressive at first, but the story quickly fell apart and left me bored to tears."
Output: {{"label": "negative"}}

Review: "It starts off a bit slow and quirky, yet turns into a deeply moving and unforgettable experience."
Output: {{"label": "positive"}}

Now classify this review:
Review:
\"\"\"{review}\"\"\"

Output:"""


def prepare_llm_text(text: str, max_chars: int = 1500) -> str:
    """
    Chỉ bỏ thẻ HTML (<br />) và chuẩn hóa khoảng trắng.
    Giữ nguyên dấu câu, viết hoa và cấu trúc câu tự nhiên cho LLM đọc hiểu.
    """
    text = HTML_TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        # Giữ phần mở đầu và phần kết (nơi người xem thường chốt đánh giá)
        half = max_chars // 2
        text = text[:half] + " ... " + text[-half:]
    return text


def call_llm_api(
    provider: str,
    model: str,
    api_key: str,
    prompt: str,
    max_retries: int = 5,
) -> tuple[str, float]:
    """Gọi API miễn phí bằng urllib chuẩn, có tự động retry khi gặp Rate Limit (429)."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "NLP-HW03-StudentClient/1.0",
    }

    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
    elif provider == "hf":
        url = "https://router.huggingface.co/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
    elif provider == "gemini":
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 64,
                "responseMimeType": "application/json",
            },
        }
    else:
        raise ValueError(f"Không hỗ trợ provider: {provider}")

    body = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw_resp = json.loads(resp.read().decode("utf-8"))
            latency = time.perf_counter() - t0

            if provider in ("groq", "hf"):
                content = (raw_resp["choices"][0]["message"].get("content") or "").strip()
            else:
                parts = raw_resp["candidates"][0].get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in parts).strip()
            return content, latency

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_s = min(60, 5 * (2 ** (attempt - 1)))
                print(f"  [WARN] HTTP {e.code} (lần {attempt}/{max_retries}). Chờ {wait_s}s rồi thử lại...")
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"Lỗi HTTP {e.code} từ {provider}: {err_body[:300]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries:
                wait_s = 5 * attempt
                print(f"  [WARN] Lỗi mạng ({e}). Chờ {wait_s}s rồi thử lại...")
                time.sleep(wait_s)
                continue
            raise

    raise RuntimeError("Vượt quá số lần thử lại API.")


def parse_llm_output(raw_text: str) -> tuple[int | None, bool]:
    """
    Ánh xạ đầu ra văn bản của LLM về nhãn số {0, 1}.
    Trả về (label, is_strict_json_valid):
      - is_strict_json_valid=True nếu đúng chuẩn JSON yêu cầu.
      - Nếu sai JSON nhưng vẫn trích được từ khóa duy nhất, trả về (label, False).
      - Nếu hoàn toàn không ánh xạ được, trả về (None, False).
    """
    cleaned = raw_text.strip()
    # Loại bỏ markdown code fence nếu có (```json ... ```)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # 1. Thử phân tích JSON chuẩn
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "label" in obj:
            val = str(obj["label"]).strip().lower()
            if val == "positive":
                return 1, True
            if val == "negative":
                return 0, True
    except json.JSONDecodeError:
        pass

    # 2. Fallback: tìm chuỗi JSON lồng trong đoạn giải thích thừa
    json_match = re.search(r'"label"\s*:\s*"(positive|negative)"', cleaned, flags=re.IGNORECASE)
    if json_match:
        return (1 if json_match.group(1).lower() == "positive" else 0), False

    # 3. Fallback cuối: chỉ khớp khi xuất hiện đúng 1 trong 2 nhãn
    has_pos = bool(re.search(r"\bpositive\b", cleaned, flags=re.IGNORECASE))
    has_neg = bool(re.search(r"\bnegative\b", cleaned, flags=re.IGNORECASE))
    if has_pos and not has_neg:
        return 1, False
    if has_neg and not has_pos:
        return 0, False

    return None, False


def run_llm_mode(
    mode: str,
    provider: str,
    model: str,
    api_key: str,
    texts: list[str],
    labels: list[int],
    delay: float,
    cache_dir: Path,
) -> dict:
    template = ZERO_SHOT_TEMPLATE if mode == "zero" else FEW_SHOT_TEMPLATE
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    cache_file = cache_dir / f"llm_cache_{provider}_{safe_model}_{mode}_{len(texts)}.jsonl"

    # Đọc lại kết quả đã lưu (resume nếu lần trước bị ngắt giữa chừng)
    cached: dict[int, dict] = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached[rec["idx"]] = rec
        print(f"[INFO] Đã khôi phục {len(cached)}/{len(texts)} mẫu từ cache: {cache_file.name}")

    print(f"\n{'#' * 65}")
    print(f"CHẠY LLM API | Provider={provider} | Model={model} | Mode={mode}-shot")
    print(f"{'#' * 65}")

    with open(cache_file, "a", encoding="utf-8") as f_out:
        for idx, (raw_text, true_y) in enumerate(zip(texts, labels)):
            if idx in cached:
                continue

            review_clean = prepare_llm_text(raw_text)
            prompt = template.format(review=review_clean)
            raw_output, latency = call_llm_api(provider, model, api_key, prompt)
            pred_y, strict_valid = parse_llm_output(raw_output)

            rec = {
                "idx": idx,
                "y_true": true_y,
                "y_pred": pred_y,
                "strict_json": strict_valid,
                "latency_s": round(latency, 4),
                "raw_output": raw_output,
            }
            cached[idx] = rec
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()

            done = len(cached)
            if done % 20 == 0 or done == len(texts):
                print(f"  Tiến độ: {done:>3}/{len(texts)} mẫu | Độ trễ gần nhất: {latency:.2f}s | Raw: {raw_output[:45]}")

            if done < len(texts) and delay > 0:
                time.sleep(delay)

    # Tổng hợp thống kê theo đúng thứ tự
    records = [cached[i] for i in range(len(texts))]
    latencies = [r["latency_s"] for r in records]
    non_strict_count = sum(1 for r in records if not r["strict_json"])
    unparseable_count = sum(1 for r in records if r["y_pred"] is None)

    valid_true = [r["y_true"] for r in records if r["y_pred"] is not None]
    valid_pred = [r["y_pred"] for r in records if r["y_pred"] is not None]
    valid_texts = [texts[r["idx"]] for r in records if r["y_pred"] is not None]

    if valid_true:
        metrics = evaluate_predictions(
            valid_true,
            valid_pred,
            f"v5 - LLM API ({provider}/{model} - {mode}-shot) [Subset {len(texts)} mẫu]",
            raw_texts=valid_texts,
            n_errors=3,
        )
    else:
        print(f"\n[WARN] Không có mẫu nào parse được nhãn hợp lệ cho {provider}/{model} ({mode}-shot).")
        metrics = {"accuracy": 0.0, "macro_f1": 0.0}

    avg_latency = sum(latencies) / len(latencies)
    print("\n--- Thống kê đặc thù cho LLM ---")
    print(f"Tỷ lệ sai chuẩn JSON (phải dùng fallback): {non_strict_count}/{len(texts)} ({non_strict_count / len(texts) * 100:.2f}%)")
    print(f"Tỷ lệ đầu ra hoàn toàn không hợp lệ      : {unparseable_count}/{len(texts)} ({unparseable_count / len(texts) * 100:.2f}%)")
    print(f"Độ trễ suy luận trung bình               : {avg_latency:.3f} s/mẫu (tổng gọi API: {sum(latencies):.1f}s)")

    return {
        "mode": f"{mode}-shot",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "invalid_rate": unparseable_count / len(texts),
        "avg_latency_s": avg_latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Bài 3 v5: Phân tích cảm xúc IMDB bằng LLM API miễn phí")
    parser.add_argument("--provider", choices=["groq", "gemini", "hf"], default="groq")
    parser.add_argument("--model", type=str, default=None, help="Tên model (để trống sẽ dùng mặc định của provider)")
    parser.add_argument("--mode", choices=["zero", "few", "both"], default="both")
    parser.add_argument("--n-samples", type=int, default=200, help="Số mẫu test phân tầng (mặc định 200 khớp với v1-v4)")
    parser.add_argument("--delay", type=float, default=None, help="Thời gian nghỉ (giây) giữa các request")
    args = parser.parse_args()

    cfg = PROVIDER_DEFAULTS[args.provider]
    model = args.model or cfg["model"]
    delay = cfg["delay"] if args.delay is None else args.delay
    api_key = os.environ.get(cfg["env_var"], "").strip()

    if not api_key:
        raise SystemExit(
            f"[LỖI] Chưa thiết lập biến môi trường {cfg['env_var']}.\n"
            f"Trong PowerShell, chạy lệnh:\n"
            f'  $env:{cfg["env_var"]}="YOUR_API_KEY_HERE"\n'
            f"rồi chạy lại: python Bai3_v5_llm.py --provider {args.provider} --mode {args.mode}"
        )

    print("[1/2] Nạp tập con kiểm thử phân tầng IMDB...")
    _, _, X_test_raw, y_test = load_imdb_data()
    X_sub, y_sub = get_stratified_subset(X_test_raw, y_test, n_samples=args.n_samples, seed=SEED)
    print(f"Số mẫu đánh giá: {len(X_sub)} (pos={sum(y_sub)}, neg={len(y_sub) - sum(y_sub)})")

    cache_dir = Path("./data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[2/2] Bắt đầu gọi LLM API...")
    modes = ["zero", "few"] if args.mode == "both" else [args.mode]
    summaries = []
    for m in modes:
        summaries.append(
            run_llm_mode(m, args.provider, model, api_key, X_sub, y_sub, delay, cache_dir)
        )

    print("\n=== TÓM TẮT KẾT QUẢ LLM API ===")
    print(f"{'Mode':<10} | {'Acc':>8} | {'Macro-F1':>8} | {'Invalid %':>10} | {'Avg Latency':>12}")
    print("-" * 58)
    for s in summaries:
        print(
            f"{s['mode']:<10} | {s['accuracy']:>8.4f} | {s['macro_f1']:>8.4f} | "
            f"{s['invalid_rate'] * 100:>9.2f}% | {s['avg_latency_s']:>10.3f}s"
        )


if __name__ == "__main__":
    main()