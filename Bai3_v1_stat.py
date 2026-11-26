"""
Bai3_v1_stat.py
Phương pháp 1: Phân loại cảm xúc IMDB bằng Thống kê thuần túy
(Tự cài đặt Multinomial Naive Bayes bằng đếm tần suất + làm trơn Laplace + Log-Likelihood)
"""

import math
import time
from collections import Counter

from imdb_utils import (
    clean_text,
    evaluate_predictions,
    get_stratified_subset,
    load_imdb_data,
)


class PureStatisticalNaiveBayes:
    """
    Bộ phân loại thống kê thuần túy dựa trên tần suất từ và định lý Bayes.
    Không sử dụng thuật toán tối ưu hay mô hình học máy có sẵn.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        use_bigrams: bool = False,
        min_freq: int = 2,
        remove_stopwords: bool = True,
    ):
        self.alpha = alpha
        self.use_bigrams = use_bigrams
        self.min_freq = min_freq
        self.remove_stopwords = remove_stopwords

        self.vocab: set[str] = set()
        self.log_prior: dict[int, float] = {}
        self.log_likelihood: dict[int, dict[str, float]] = {0: {}, 1: {}}
        self.unseen_in_class_log_prob: dict[int, float] = {}

    def _extract_features(self, text: str) -> list[str]:
        """Làm sạch văn bản và tách thành danh sách đặc trưng (unigram / unigram + bigram)."""
        cleaned = clean_text(text, remove_stopwords=self.remove_stopwords)
        tokens = cleaned.split()
        if not self.use_bigrams:
            return tokens

        bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit(self, texts: list[str], labels: list[int]) -> "PureStatisticalNaiveBayes":
        """Ước lượng P(c) và P(w|c) bằng cách đếm tần suất trực tiếp trên tập huấn luyện."""
        total_docs = len(labels)
        doc_counts = Counter(labels)
        word_counts = {0: Counter(), 1: Counter()}
        global_word_counts = Counter()

        for text, label in zip(texts, labels):
            features = self._extract_features(text)
            word_counts[label].update(features)
            global_word_counts.update(features)

        self.vocab = {w for w, cnt in global_word_counts.items() if cnt >= self.min_freq}
        vocab_size = len(self.vocab)

        for c in (0, 1):
            self.log_prior[c] = math.log(doc_counts[c] / total_docs)

            total_tokens_in_c = sum(cnt for w, cnt in word_counts[c].items() if w in self.vocab)
            denom = total_tokens_in_c + self.alpha * vocab_size

            self.unseen_in_class_log_prob[c] = math.log(self.alpha / denom)

            self.log_likelihood[c] = {
                w: math.log((cnt + self.alpha) / denom)
                for w, cnt in word_counts[c].items()
                if w in self.vocab
            }

        return self

    def predict_one(self, text: str) -> int:
        """Tính tổng log-xác suất hậu nghiệm cho mỗi lớp và chọn lớp lớn hơn."""
        features = self._extract_features(text)
        scores = {0: self.log_prior[0], 1: self.log_prior[1]}

        for feat in features:
            if feat not in self.vocab:
                continue
            scores[0] += self.log_likelihood[0].get(feat, self.unseen_in_class_log_prob[0])
            scores[1] += self.log_likelihood[1].get(feat, self.unseen_in_class_log_prob[1])

        return 1 if scores[1] >= scores[0] else 0

    def predict(self, texts: list[str]) -> list[int]:
        return [self.predict_one(t) for t in texts]

    def get_top_discriminative_features(self, top_k: int = 10) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Tính tỷ số log-likelihood ratio: log P(w|pos) - log P(w|neg)
        để minh họa khả năng giải thích cao của mô hình thống kê.
        """
        ratios = []
        for w in self.vocab:
            lp_pos = self.log_likelihood[1].get(w, self.unseen_in_class_log_prob[1])
            lp_neg = self.log_likelihood[0].get(w, self.unseen_in_class_log_prob[0])
            ratios.append((w, lp_pos - lp_neg))

        ratios.sort(key=lambda x: x[1], reverse=True)
        top_pos = ratios[:top_k]
        top_neg = ratios[-top_k:][::-1]
        return top_pos, top_neg


def main():
    print("[1/3] Đang nạp dữ liệu IMDB...")
    X_train, y_train, X_test, y_test = load_imdb_data()
    X_sub, y_sub = get_stratified_subset(X_test, y_test, n_samples=200, seed=42)

    configs = [
        ("v1a - Thống kê thuần túy (Naive Bayes Unigram, Laplace alpha=1)", False),
        ("v1b - Thống kê thuần túy (Naive Bayes Unigram + Bigram, Laplace alpha=1)", True),
    ]

    for name, use_bigrams in configs:
        print(f"\n[2/3] Huấn luyện & suy luận: {name}...")
        t0 = time.time()

        model = PureStatisticalNaiveBayes(
            alpha=1.0,
            use_bigrams=use_bigrams,
            min_freq=2,
            remove_stopwords=True,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - t0

        print(f"Kích thước từ vựng |V|: {len(model.vocab):,} | Tổng thời gian: {elapsed:.2f}s")
        evaluate_predictions(y_test, y_pred, name, raw_texts=X_test, n_errors=3)

        y_sub_pred = model.predict(X_sub)
        sub_metrics = evaluate_predictions(
            y_sub, y_sub_pred, f"{name} [Subset 200 mẫu]", raw_texts=None, n_errors=0
        )

        if use_bigrams:
            top_pos, top_neg = model.get_top_discriminative_features(top_k=10)
            print("\n[3/3] Top 10 đặc trưng nghiêng về TÍCH CỰC nhất (log P(w|pos) - log P(w|neg)):")
            for w, score in top_pos:
                print(f"  + {w:<22s} : {score:+.3f}")
            print("\nTop 10 đặc trưng nghiêng về TIÊU CỰC nhất:")
            for w, score in top_neg:
                print(f"  - {w:<22s} : {score:+.3f}")


if __name__ == "__main__":
    main()