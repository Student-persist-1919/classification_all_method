"""
imdb_utils.py
Module dùng chung cho Bài 3 (v1 -> v5):
- Tải và giải nén bộ dữ liệu IMDB (aclImdb_v1.tar.gz)
- Đọc dữ liệu train/test (25.000 mẫu mỗi tập)
- Làm sạch văn bản bằng Regex (xử lý HTML, viết tắt phủ định, ký tự nhiễu)
- Lấy mẫu phân tầng cố định (dùng chung để so sánh công bằng với LLM API ở v5)
- Đánh giá mô hình (Accuracy, Macro-F1, Confusion Matrix, Phân tích lỗi)
"""

import random
import re
import tarfile
import urllib.request
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DEFAULT_DATA_DIR = Path("./data")

NEGATION_WORDS = {
    "no", "not", "nor", "never", "neither", "none",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
}
SENTIMENT_STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'m\b"), " am"),
]
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
WHITESPACE_RE = re.compile(r"\s+")


def download_and_extract_imdb(data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Tải và giải nén tập dữ liệu IMDB nếu chưa tồn tại."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "aclImdb_v1.tar.gz"
    extract_dir = data_dir / "aclImdb"

    if not extract_dir.exists():
        if not archive_path.exists():
            print(f"[INFO] Đang tải dữ liệu IMDB từ {DATA_URL} (~84 MB)...")
            urllib.request.urlretrieve(DATA_URL, archive_path)
            print("[INFO] Tải xong!")

        print("[INFO] Đang giải nén aclImdb_v1.tar.gz...")
        with tarfile.open(archive_path, "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=data_dir, filter="data")
            else:
                tar.extractall(path=data_dir)
        print("[INFO] Giải nén hoàn tất!")

    return extract_dir


def _load_split(split_dir: Path) -> tuple[list[str], list[int]]:
    """Đọc toàn bộ file .txt từ thư mục neg (0) và pos (1)."""
    texts, labels = [], []
    for folder_name, label in (("neg", 0), ("pos", 1)):
        folder = split_dir / folder_name
        for file_path in sorted(folder.glob("*.txt")):
            texts.append(file_path.read_text(encoding="utf-8"))
            labels.append(label)
    return texts, labels


def load_imdb_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Trả về (X_train_raw, y_train, X_test_raw, y_test)."""
    extract_dir = download_and_extract_imdb(data_dir)
    X_train, y_train = _load_split(extract_dir / "train")
    X_test, y_test = _load_split(extract_dir / "test")
    return X_train, y_train, X_test, y_test


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Làm sạch văn bản nhận xét phim:
    1. Bỏ thẻ HTML (<br />, ...) và URL
    2. Chuyển chữ thường và chuẩn hóa dạng viết tắt phủ định (don't -> do not)
    3. Bỏ dấu câu, chữ số, ký tự đặc biệt
    4. Tùy chọn loại bỏ stopword (nhưng vẫn giữ từ phủ định như not, no, never)
    """
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    for pattern, repl in CONTRACTIONS:
        text = pattern.sub(repl, text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if remove_stopwords:
        tokens = [w for w in text.split() if len(w) > 1 and w not in SENTIMENT_STOPWORDS]
        return " ".join(tokens)

    return text


def get_stratified_subset(
    texts: list[str],
    labels: list[int],
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Lấy tập con cân bằng 50% neg / 50% pos với seed cố định.
    Dùng để so sánh công bằng cả 5 phương pháp với LLM API (v5).
    """
    rng = random.Random(seed)
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    pos_idx = [i for i, y in enumerate(labels) if y == 1]

    half = n_samples // 2
    selected = rng.sample(neg_idx, half) + rng.sample(pos_idx, half)
    rng.shuffle(selected)

    sub_texts = [texts[i] for i in selected]
    sub_labels = [labels[i] for i in selected]
    return sub_texts, sub_labels


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    model_name: str,
    raw_texts: list[str] | None = None,
    n_errors: int = 3,
) -> dict[str, float]:
    """In báo cáo đánh giá chuẩn và trả về Accuracy, Macro-F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 65}")
    print(f"KẾT QUẢ ĐÁNH GIÁ: {model_name}")
    print(f"{'=' * 65}")
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Tiêu cực (0)", "Tích cực (1)"],
            digits=4,
        )
    )
    print("Confusion Matrix (Hàng: Thực tế | Cột: Dự đoán):")
    print(f"              Dự đoán 0   Dự đoán 1")
    print(f"Thực tế 0  : {cm[0, 0]:>9,}   {cm[0, 1]:>9,}")
    print(f"Thực tế 1  : {cm[1, 0]:>9,}   {cm[1, 1]:>9,}")

    if raw_texts is not None and n_errors > 0:
        wrong_idx = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
        print(f"\nTổng số mẫu sai: {len(wrong_idx):,} / {len(y_true):,}")
        print(f"--- Ví dụ {min(n_errors, len(wrong_idx))} mẫu phân loại sai ---")
        for idx in wrong_idx[:n_errors]:
            snippet = clean_text(raw_texts[idx])[:180]
            print(f"  [#{idx}] Thực tế={y_true[idx]} | Dự đoán={y_pred[idx]} | \"{snippet}...\"")

    return {"accuracy": acc, "macro_f1": macro_f1}