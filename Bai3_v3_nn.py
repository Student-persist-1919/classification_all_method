"""
Bai3_v3_nn.py
Phương pháp 3: Phân loại cảm xúc IMDB bằng Mạng nơ-ron truyền thẳng (MLP - PyTorch)
- Đầu vào: vector TF-IDF (unigram + bigram, giới hạn 20.000 đặc trưng giàu thông tin nhất)
- Kiến trúc: Input(20000) -> Linear(256) -> LayerNorm -> ReLU -> Dropout(0.5)
             -> Linear(64) -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear(1)
- Hàm mất mát: BCEWithLogitsLoss (Binary Cross-Entropy tích hợp Sigmoid ổn định số học)
- Tối ưu: AdamW (kèm L2 weight decay) + Early Stopping dựa trên Dev Loss
- Tối ưu RAM: chuyển đổi ma trận thưa sang dense theo từng mini-batch trong collate_fn
"""

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from imdb_utils import (
    SENTIMENT_STOPWORDS,
    clean_text,
    evaluate_predictions,
    get_stratified_subset,
    load_imdb_data,
)

SEED = 42
MAX_FEATURES = 20_000
BATCH_SIZE = 128
EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

STOPWORDS_LIST = sorted(SENTIMENT_STOPWORDS)


def set_seed(seed: int = SEED) -> None:
    """Cố định seed để kết quả huấn luyện có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class IndexDataset(Dataset):
    """Dataset nhẹ chỉ trả về chỉ số mẫu để cắt ma trận thưa theo từng batch."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> int:
        return idx


def make_sparse_loader(
    X_sparse,
    y: list[int] | None = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> DataLoader:
    """
    Tạo DataLoader chuyển đổi ma trận CSR thưa sang Tensor dày theo từng mini-batch.
    Giúp tiết kiệm RAM (chỉ ~10 MB/batch thay vì nạp toàn bộ 2 GB ma trận dày).
    """

    def collate_fn(indices: list[int]):
        xb = torch.from_numpy(X_sparse[indices].toarray().astype(np.float32))
        if y is None:
            return xb
        yb = torch.tensor([y[i] for i in indices], dtype=torch.float32)
        return xb, yb

    return DataLoader(
        IndexDataset(X_sparse.shape[0]),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


class SentimentMLP(nn.Module):
    """Mạng nơ-ron truyền thẳng 2 lớp ẩn cho phân loại cảm xúc nhị phân."""

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 256,
        hidden2: int = 64,
        dropout1: float = 0.5,
        dropout2: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_n = yb.size(0)
        total_loss += loss.item() * batch_n
        total_samples += batch_n

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
) -> tuple[float | None, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item() * yb.size(0)
                total_samples += yb.size(0)
            y_true_all.extend(yb.long().cpu().tolist())
        else:
            xb = batch.to(device)
            logits = model(xb)

        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
        y_pred_all.extend(preds)

    avg_loss = (total_loss / total_samples) if (criterion is not None and total_samples > 0) else None
    return avg_loss, y_true_all, y_pred_all


def main():
    set_seed(SEED)
    device = get_device()
    print(f"[INFO] Thiết bị huấn luyện: {device}")

    print("[1/4] Đang nạp và làm sạch dữ liệu IMDB...")
    t0 = time.time()
    X_train_raw, y_train, X_test_raw, y_test = load_imdb_data()
    X_sub_raw, y_sub = get_stratified_subset(X_test_raw, y_test, n_samples=200, seed=SEED)

    X_train_clean = [clean_text(t, remove_stopwords=False) for t in X_train_raw]
    X_test_clean = [clean_text(t, remove_stopwords=False) for t in X_test_raw]
    X_sub_clean = [clean_text(t, remove_stopwords=False) for t in X_sub_raw]

    X_tr, X_dev, y_tr, y_dev = train_test_split(
        X_train_clean,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=SEED,
    )
    print(
        f"Hoàn tất tiền xử lý ({time.time() - t0:.2f}s) | "
        f"Train={len(X_tr):,} | Dev={len(X_dev):,} | Test={len(X_test_clean):,}"
    )

    print("\n[2/4] Trích xuất đặc trưng TF-IDF (fit duy nhất trên tập Train)...")
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS_LIST,
        max_features=MAX_FEATURES,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=False,
    )
    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_dev_vec = vectorizer.transform(X_dev)
    X_test_vec = vectorizer.transform(X_test_clean)
    X_sub_vec = vectorizer.transform(X_sub_clean)
    print(f"Số chiều đầu vào mạng MLP: {X_tr_vec.shape[1]:,}")

    train_loader = make_sparse_loader(X_tr_vec, y_tr, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = make_sparse_loader(X_dev_vec, y_dev, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_sparse_loader(X_test_vec, y_test, batch_size=BATCH_SIZE, shuffle=False)
    sub_loader = make_sparse_loader(X_sub_vec, y_sub, batch_size=BATCH_SIZE, shuffle=False)

    print("\n[3/4] Huấn luyện mạng nơ-ron MLP...")
    model = SentimentMLP(input_dim=X_tr_vec.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_dev_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Dev Loss':>10} | {'Dev Acc':>8} | {'Dev F1':>8} | {'Time':>6}")
    print("-" * 62)

    t_train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_true, dev_pred = evaluate_loader(model, dev_loader, criterion, device)

        dev_acc = accuracy_score(dev_true, dev_pred)
        dev_f1 = f1_score(dev_true, dev_pred, average="macro")
        scheduler.step(dev_loss)

        marker = ""
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1

        print(
            f"{epoch:>5} | {train_loss:>10.4f} | {dev_loss:>10.4f} | "
            f"{dev_acc:>8.4f} | {dev_f1:>8.4f} | {time.time() - t_ep:>5.1f}s{marker}"
        )

        if no_improve >= PATIENCE:
            print(f"[Early Stopping] Dừng sớm tại epoch {epoch} (tốt nhất tại epoch {best_epoch}).")
            break

    print(f"Tổng thời gian huấn luyện: {time.time() - t_train_start:.2f}s")

    model.load_state_dict(best_state)

    print("\n[4/4] Đánh giá mô hình MLP tốt nhất trên tập kiểm thử IMDB...")
    _, _, y_test_pred = evaluate_loader(model, test_loader, criterion=None, device=device)
    _, _, y_sub_pred = evaluate_loader(model, sub_loader, criterion=None, device=device)

    evaluate_predictions(
        y_test,
        y_test_pred,
        f"v3 - Mạng nơ-ron MLP (PyTorch, TF-IDF 20k, best epoch={best_epoch})",
        raw_texts=X_test_raw,
        n_errors=3,
    )
    evaluate_predictions(
        y_sub,
        y_sub_pred,
        "v3 - Mạng nơ-ron MLP [Subset 200 mẫu]",
        raw_texts=None,
        n_errors=0,
    )


if __name__ == "__main__":
    main()