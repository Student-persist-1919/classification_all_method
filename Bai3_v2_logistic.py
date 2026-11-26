"""
Bai3_v2_logistic.py
Phương pháp 2: Phân loại cảm xúc IMDB bằng Mô hình học máy (Logistic Regression)
- Tái sử dụng dữ liệu và hàm làm sạch từ imdb_utils.py
- So sánh 3 biểu diễn đặc trưng: BoW, TF-IDF, TF-IDF + Bigram (1, 2)
- Chọn hệ số điều chuẩn C (nghịch đảo của alpha) trên tập dev (20% tập train)
- Huấn luyện lại trên toàn bộ 25.000 mẫu train và đánh giá trên tập test (25.000 mẫu)
  cũng như tập con 200 mẫu cố định để đối chiếu với v1 -> v5
"""

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from imdb_utils import (
    SENTIMENT_STOPWORDS,
    clean_text,
    evaluate_predictions,
    get_stratified_subset,
    load_imdb_data,
)

STOPWORDS_LIST = sorted(SENTIMENT_STOPWORDS)
C_CANDIDATES = [0.2, 0.5, 1.0, 2.0, 5.0]


def make_vectorizer(kind: str):
    """Khởi tạo bộ trích xuất đặc trưng theo từng thí nghiệm."""
    if kind == "bow":
        return CountVectorizer(
            stop_words=STOPWORDS_LIST,
            min_df=2,
            lowercase=False,
        )
    if kind == "tfidf":
        return TfidfVectorizer(
            stop_words=STOPWORDS_LIST,
            min_df=2,
            lowercase=False,
        )
    if kind == "tfidf_bigram":
        return TfidfVectorizer(
            stop_words=STOPWORDS_LIST,
            min_df=2,
            ngram_range=(1, 2),
            lowercase=False,
        )
    raise ValueError(f"Không hỗ trợ loại vectorizer: {kind}")


def select_best_c(
    kind: str,
    X_tr: list[str],
    y_tr: list[int],
    X_dev: list[str],
    y_dev: list[int],
) -> tuple[float, float]:
    """Chọn siêu tham số C tốt nhất trên tập dev (không chạm vào tập test)."""
    vec = make_vectorizer(kind)
    X_tr_vec = vec.fit_transform(X_tr)
    X_dev_vec = vec.transform(X_dev)

    best_c = C_CANDIDATES[0]
    best_dev_f1 = -1.0

    for c in C_CANDIDATES:
        clf = LogisticRegression(
            C=c,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_tr_vec, y_tr)
        dev_pred = clf.predict(X_dev_vec)
        dev_f1 = f1_score(y_dev, dev_pred, average="macro")
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_c = c

    return best_c, best_dev_f1


def print_top_weights(vectorizer, clf: LogisticRegression, top_k: int = 10) -> None:
    """In các đặc trưng có trọng số w lớn nhất và nhỏ nhất để giải thích mô hình."""
    feature_names = vectorizer.get_feature_names_out()
    weights = clf.coef_[0]

    top_pos_idx = np.argsort(weights)[-top_k:][::-1]
    top_neg_idx = np.argsort(weights)[:top_k]

    print(f"\nTop {top_k} đặc trưng có trọng số DƯƠNG lớn nhất (nghiêng về TÍCH CỰC):")
    for idx in top_pos_idx:
        print(f"  + {feature_names[idx]:<24s} : {weights[idx]:+.4f}")

    print(f"\nTop {top_k} đặc trưng có trọng số ÂM lớn nhất (nghiêng về TIÊU CỰC):")
    for idx in top_neg_idx:
        print(f"  - {feature_names[idx]:<24s} : {weights[idx]:+.4f}")


def main():
    print("[1/4] Đang nạp và làm sạch dữ liệu IMDB...")
    t_load = time.time()
    X_train_raw, y_train, X_test_raw, y_test = load_imdb_data()
    X_sub_raw, y_sub = get_stratified_subset(X_test_raw, y_test, n_samples=200, seed=42)

    X_train_clean = [clean_text(t, remove_stopwords=False) for t in X_train_raw]
    X_test_clean = [clean_text(t, remove_stopwords=False) for t in X_test_raw]
    X_sub_clean = [clean_text(t, remove_stopwords=False) for t in X_sub_raw]
    print(f"Hoàn tất tiền xử lý 50.000 mẫu trong {time.time() - t_load:.2f}s.")

    X_tr, X_dev, y_tr, y_dev = train_test_split(
        X_train_clean,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    experiments = [
        ("v2a - BoW + Logistic Regression", "bow"),
        ("v2b - TF-IDF + Logistic Regression", "tfidf"),
        ("v2c - TF-IDF + Bigram (1,2) + Logistic Regression", "tfidf_bigram"),
    ]

    summary_rows = []
    best_artifact = None

    print("\n[2/4] Chọn siêu tham số trên dev và huấn luyện mô hình cuối trên full train...")
    for exp_name, kind in experiments:
        t0 = time.time()

        best_c, dev_f1 = select_best_c(kind, X_tr, y_tr, X_dev, y_dev)

        vectorizer = make_vectorizer(kind)
        X_train_vec = vectorizer.fit_transform(X_train_clean)
        X_test_vec = vectorizer.transform(X_test_clean)
        X_sub_vec = vectorizer.transform(X_sub_clean)

        clf = LogisticRegression(
            C=best_c,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_train_vec, y_train)

        y_pred = clf.predict(X_test_vec)
        y_sub_pred = clf.predict(X_sub_vec)

        acc_full = accuracy_score(y_test, y_pred)
        f1_full = f1_score(y_test, y_pred, average="macro")
        acc_sub = accuracy_score(y_sub, y_sub_pred)
        f1_sub = f1_score(y_sub, y_sub_pred, average="macro")
        elapsed = time.time() - t0

        summary_rows.append(
            {
                "Thí nghiệm": exp_name,
                "Best C": best_c,
                "Dev F1": round(dev_f1, 4),
                "Số đặc trưng": X_train_vec.shape[1],
                "Test Acc": round(acc_full, 4),
                "Test Macro-F1": round(f1_full, 4),
                "Sub200 Acc": round(acc_sub, 4),
                "Sub200 F1": round(f1_sub, 4),
                "Thời gian (s)": round(elapsed, 2),
            }
        )
        print(
            f"  -> {exp_name} | Best C={best_c} | Features={X_train_vec.shape[1]:,} | "
            f"Test Acc={acc_full:.4f} | Test F1={f1_full:.4f} ({elapsed:.2f}s)"
        )

        if best_artifact is None or f1_full > best_artifact["f1_full"]:
            best_artifact = {
                "name": exp_name,
                "best_c": best_c,
                "f1_full": f1_full,
                "vectorizer": vectorizer,
                "clf": clf,
                "y_pred": y_pred,
                "y_sub_pred": y_sub_pred,
            }

    print("\n[3/4] BẢNG TỔNG HỢP KẾT QUẢ PHƯƠNG PHÁP 2 (LOGISTIC REGRESSION)")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n[4/4] ĐÁNH GIÁ CHI TIẾT & PHÂN TÍCH TRỌNG SỐ MÔ HÌNH TỐT NHẤT")
    evaluate_predictions(
        y_test,
        best_artifact["y_pred"].tolist(),
        f"{best_artifact['name']} (C={best_artifact['best_c']})",
        raw_texts=X_test_raw,
        n_errors=3,
    )
    evaluate_predictions(
        y_sub,
        best_artifact["y_sub_pred"].tolist(),
        f"{best_artifact['name']} [Subset 200 mẫu]",
        raw_texts=None,
        n_errors=0,
    )
    print_top_weights(best_artifact["vectorizer"], best_artifact["clf"], top_k=10)


if __name__ == "__main__":
    main()