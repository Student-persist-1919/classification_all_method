"""
Bai3_v4_cnn_rnn.py
Phương pháp 4: Phân loại cảm xúc IMDB bằng Mạng CNN (TextCNN) và RNN (BiGRU) - PyTorch
- Tái sử dụng dữ liệu và hàm làm sạch từ imdb_utils.py (giữ nguyên stopword để bảo toàn cấu trúc câu)
- Hỗ trợ chạy TextCNN, BiGRU hoặc cả hai:
    python Bai3_v4_cnn_rnn.py --arch cnn
    python Bai3_v4_cnn_rnn.py --arch rnn
    python Bai3_v4_cnn_rnn.py --arch both
"""

import argparse
import copy
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from imdb_utils import (
    clean_text,
    evaluate_predictions,
    get_stratified_subset,
    load_imdb_data,
)

# Cấu hình siêu tham số tối ưu cho cả CPU lẫn GPU
SEED = 42
MAX_VOCAB_SIZE = 25_000
MIN_FREQ = 2
MAX_LEN = 256
EMBED_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
PATIENCE = 2
LEARNING_RATE = 1e-3

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def set_seed(seed: int = SEED) -> None:
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


def build_vocab(
    texts: list[str],
    max_vocab_size: int = MAX_VOCAB_SIZE,
    min_freq: int = MIN_FREQ,
) -> dict[str, int]:
    """Xây dựng từ điển chỉ từ tập train (tránh rò rỉ dữ liệu)."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, freq in counter.most_common(max_vocab_size - 2):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    """Ánh xạ chuỗi từ sang chuỗi chỉ số (đảm bảo độ dài tối thiểu 1)."""
    tokens = text.split()[:max_len]
    ids = [vocab.get(tok, UNK_IDX) for tok in tokens]
    return ids if ids else [UNK_IDX]


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.sequences[idx], self.labels[idx]


def collate_sequences(batch: list[tuple[list[int], int]]):
    """
    Dynamic padding theo độ dài lớn nhất trong từng batch (tối thiểu 5 cho kernel CNN).
    Giúp tăng tốc đáng kể khi huấn luyện trên CPU.
    """
    seqs, labels = zip(*batch)
    lengths = [len(s) for s in seqs]
    batch_max_len = max(5, max(lengths))

    padded = torch.full((len(seqs), batch_max_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)

    return (
        padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


class TextCNN(nn.Module):
    """
    Mô hình TextCNN (Kim, 2014):
    - Lớp Embedding học biểu diễn dày cho từng từ
    - Nhiều bộ lọc Conv1d (kernel 3, 4, 5) phát hiện cụm n-gram đặc trưng
    - Global Max-Pooling giữ tín hiệu mạnh nhất của từng bộ lọc
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embed_drop = nn.Dropout(0.3)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> (B, L, E) -> (B, E, L) cho Conv1d
        emb = self.embed_drop(self.embedding(x)).transpose(1, 2)
        pooled = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat)).squeeze(-1)


class BiGRUClassifier(nn.Module):
    """
    Mô hình RNN hai chiều (Bidirectional GRU):
    - Đọc chuỗi từ theo cả hai chiều trái -> phải và phải -> trái
    - Dùng pack_padded_sequence để bỏ qua các token <PAD>
    - Ghép trạng thái ẩn cuối cùng của hai chiều làm biểu diễn văn bản
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embed_drop = nn.Dropout(0.3)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed_drop(self.embedding(x))
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        # h_n[-2]: trạng thái cuối chiều xuôi; h_n[-1]: trạng thái cuối chiều ngược
        doc_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(doc_repr)).squeeze(-1)


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

    for xb, lengths, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb, lengths)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * yb.size(0)
        total_samples += yb.size(0)

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
    y_true_all, y_pred_all = [], []

    for xb, lengths, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb, lengths)
        if criterion is not None:
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
        y_true_all.extend(yb.long().cpu().tolist())
        y_pred_all.extend(preds)

    avg_loss = (total_loss / total_samples) if (criterion and total_samples > 0) else None
    return avg_loss, y_true_all, y_pred_all


def run_experiment(
    arch_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    sub_loader: DataLoader,
    X_test_raw: list[str],
    device: torch.device,
) -> dict:
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'#' * 65}")
    print(f"HUẤN LUYỆN: {arch_name} | Số tham số: {n_params:,}")
    print(f"{'#' * 65}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Dev Loss':>10} | {'Dev Acc':>8} | {'Dev F1':>8} | {'Time':>6}")
    print("-" * 62)

    best_dev_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    t_start = time.time()

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
            print(f"[Early Stopping] Dừng sớm tại epoch {epoch} (tốt nhất: epoch {best_epoch}).")
            break

    train_time = time.time() - t_start
    model.load_state_dict(best_state)

    _, y_test_true, y_test_pred = evaluate_loader(model, test_loader, None, device)
    _, y_sub_true, y_sub_pred = evaluate_loader(model, sub_loader, None, device)

    full_m = evaluate_predictions(
        y_test_true,
        y_test_pred,
        f"{arch_name} (best epoch={best_epoch})",
        raw_texts=X_test_raw,
        n_errors=3,
    )
    sub_m = evaluate_predictions(
        y_sub_true,
        y_sub_pred,
        f"{arch_name} [Subset 200 mẫu]",
        raw_texts=None,
        n_errors=0,
    )

    return {
        "Mô hình": arch_name,
        "Best Epoch": best_epoch,
        "Test Acc": round(full_m["accuracy"], 4),
        "Test Macro-F1": round(full_m["macro_f1"], 4),
        "Sub200 Acc": round(sub_m["accuracy"], 4),
        "Sub200 F1": round(sub_m["macro_f1"], 4),
        "Thời gian (s)": round(train_time, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Bài 3 v4: TextCNN & BiGRU trên IMDB")
    parser.add_argument(
        "--arch",
        choices=["cnn", "rnn", "both"],
        default="both",
        help="Chọn kiến trúc cần chạy (mặc định: both)",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"[INFO] Thiết bị huấn luyện: {device}")

    print("[1/3] Nạp và tiền xử lý chuỗi văn bản (giữ nguyên thứ tự từ & stopword)...")
    t0 = time.time()
    X_train_raw, y_train, X_test_raw, y_test = load_imdb_data()
    X_sub_raw, y_sub = get_stratified_subset(X_test_raw, y_test, n_samples=200, seed=SEED)

    X_train_clean = [clean_text(t, remove_stopwords=False) for t in X_train_raw]
    X_test_clean = [clean_text(t, remove_stopwords=False) for t in X_test_raw]
    X_sub_clean = [clean_text(t, remove_stopwords=False) for t in X_sub_raw]

    X_tr, X_dev, y_tr, y_dev = train_test_split(
        X_train_clean, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    print("[2/3] Xây dựng từ điển và mã hóa chuỗi token...")
    vocab = build_vocab(X_tr)
    print(f"Kích thước từ vựng: {len(vocab):,} | MAX_LEN={MAX_LEN} | Chuẩn bị: {time.time() - t0:.2f}s")

    tr_seqs = [encode_text(t, vocab) for t in X_tr]
    dev_seqs = [encode_text(t, vocab) for t in X_dev]
    test_seqs = [encode_text(t, vocab) for t in X_test_clean]
    sub_seqs = [encode_text(t, vocab) for t in X_sub_clean]

    train_loader = DataLoader(
        SequenceDataset(tr_seqs, y_tr), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences
    )
    dev_loader = DataLoader(
        SequenceDataset(dev_seqs, y_dev), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences
    )
    test_loader = DataLoader(
        SequenceDataset(test_seqs, y_test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences
    )
    sub_loader = DataLoader(
        SequenceDataset(sub_seqs, y_sub), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences
    )

    print("[3/3] Bắt đầu huấn luyện và đánh giá...")
    summary = []
    if args.arch in ("cnn", "both"):
        set_seed(SEED)
        summary.append(
            run_experiment(
                "v4a - TextCNN (Conv1d k=3,4,5 + MaxPool)",
                TextCNN(vocab_size=len(vocab)),
                train_loader, dev_loader, test_loader, sub_loader, X_test_raw, device,
            )
        )

    if args.arch in ("rnn", "both"):
        set_seed(SEED)
        summary.append(
            run_experiment(
                "v4b - BiGRU (Bidirectional RNN)",
                BiGRUClassifier(vocab_size=len(vocab)),
                train_loader, dev_loader, test_loader, sub_loader, X_test_raw, device,
            )
        )

    print("\n=== BẢNG TỔNG HỢP PHƯƠNG PHÁP 4 (CNN / RNN) ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()