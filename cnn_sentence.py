"""
Kim (2014) Convolutional Neural Networks for Sentence Classification —— 最简复现
数据集：MR（电影评论文本，二分类情感）
词向量：GloVe 6B 100d（原论文用 word2vec 300d，此处为便于下载改用 100d）
用法：  python cnn_sentence.py --variant rand static nonstatic multichannel --folds 10
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
GLOVE = os.path.join(DATA, "glove.6B.100d.txt")

VARIANT_NAMES = {
    "rand": "CNN-rand",
    "static": "CNN-static",
    "nonstatic": "CNN-non-static",
    "multichannel": "CNN-multichannel",
}
PAPER_MR = {"rand": 76.1, "static": 81.0, "nonstatic": 81.5, "multichannel": 81.1}


# ---------------------------------------------------------------- 数据
def load_mr():
    texts, labels = [], []
    for name, label in (("rt-polarity.pos", 1), ("rt-polarity.neg", 0)):
        path = os.path.join(DATA, name)
        with open(path, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line.split())
                    labels.append(label)
    return texts, np.array(labels, dtype=np.int64)


def build_vocab(token_lists, min_freq=1):
    freq = {}
    for tokens in token_lists:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
    words = [w for w, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_freq]
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def load_glove(word2idx, dim):
    """返回 (词表大小, dim) 的 float32 矩阵；<pad> 为零向量，未命中的词用 U(-0.25, 0.25)。"""
    rng = np.random.RandomState(1234)
    emb = rng.uniform(-0.25, 0.25, (len(word2idx), dim)).astype(np.float32)
    emb[0] = 0.0
    hit = 0
    total = 0
    with open(GLOVE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            w = parts[0]
            idx = word2idx.get(w)
            if idx is None:
                continue
            if len(parts) - 1 != dim:
                continue
            emb[idx] = np.asarray(parts[1:], dtype=np.float32)
            hit += 1
    total = len(word2idx) - 2
    print(f"  GloVe 命中 {hit}/{total} 个词 ({hit/max(1,total)*100:.1f}%)")
    return emb


def encode(token_lists, word2idx, max_len):
    x = np.zeros((len(token_lists), max_len), dtype=np.int64)
    for i, tokens in enumerate(token_lists):
        ids = [word2idx.get(t, 1) for t in tokens][:max_len]
        x[i, :len(ids)] = ids
    return x


# ---------------------------------------------------------------- 模型
class TextCNN(nn.Module):
    def __init__(self, emb_weight, variant, num_classes=2, num_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        emb_weight = torch.as_tensor(emb_weight, dtype=torch.float32)
        vocab_size, dim = emb_weight.shape
        self.variant = variant

        if variant == "multichannel":
            # 通道 1 冻结、通道 2 可微调
            self.emb_static = nn.Embedding.from_pretrained(emb_weight, freeze=True)
            self.emb_tuned = nn.Embedding.from_pretrained(emb_weight.clone(), freeze=False)
            in_ch = 2
        else:
            freeze = variant == "static"
            self.emb_static = nn.Embedding.from_pretrained(emb_weight, freeze=freeze)
            self.emb_tuned = None
            in_ch = 1

        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, num_filters, (h, dim)) for h in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        if self.variant == "multichannel":
            e = torch.stack([self.emb_static(x), self.emb_tuned(x)], dim=1)  # (B,2,L,D)
        else:
            e = self.emb_static(x).unsqueeze(1)                              # (B,1,L,D)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(e)).squeeze(-1)        # (B, F, L-h+1)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # 最大时间池化 (B, F)
            feats.append(h)
        z = self.dropout(torch.cat(feats, dim=1))
        return self.fc(z)

    def weights(self):
        """需要做 L2 约束的参数（卷积核与全连接层）。"""
        return [p for p in list(self.convs.parameters()) + list(self.fc.parameters()) if p.requires_grad]


def l2_constrain(model, s=3.0):
    with torch.no_grad():
        for p in model.weights():
            norm = p.norm(2)
            if norm > s:
                p.mul_(s / norm)


# ---------------------------------------------------------------- 训练
def train_one(x_tr, y_tr, x_te, y_te, emb, variant, args, device):
    model = TextCNN(emb, variant, num_filters=args.filters,
                    kernel_sizes=tuple(args.kernel), dropout=args.dropout).to(device)
    if args.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, eps=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_te_t = torch.from_numpy(x_te).to(device)
    n = len(x_tr)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total, seen = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(x_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()
            l2_constrain(model)      # 每步后约束权重 L2 范数上界 s=3
            total += float(loss) * len(idx)
            seen += len(idx)
        if args.verbose:
            print(f"      epoch {epoch+1}/{args.epochs} train_loss={total/seen:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(x_te_t).argmax(dim=1).cpu().numpy()
    return float((pred == y_te).mean())


def run_variant(variant, texts, labels, word2idx, emb, args, device):
    x = encode(texts, word2idx, args.max_len)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    accs = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(skf.split(x, labels), 1):
        acc = train_one(x[tr], labels[tr], x[te], labels[te], emb, variant, args, device)
        accs.append(acc)
        print(f"    fold {fold}/{args.folds}: {acc*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    return accs


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=["rand", "static", "nonstatic", "multichannel"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--filters", type=int, default=100)
    ap.add_argument("--kernel", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--max_len", type=int, default=56)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opt", choices=["adadelta", "adam"], default="adadelta")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results.json"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)

    print("=== 配置 ===")
    print(f"device={device}  variant={args.variant}  folds={args.folds}  epochs={args.epochs}")
    print(f"dim={args.dim}  filters={args.filters}  kernel={args.kernel}  "
          f"batch={args.batch}  lr={args.lr}  dropout={args.dropout}")

    texts, labels = load_mr()
    print(f"\n=== 数据 ===\nMR: {len(texts)} 句, 正类 {int(labels.sum())}, 负类 {int((1-labels).sum())}")
    word2idx = build_vocab(texts)
    lengths = [len(t) for t in texts]
    print(f"词表 {len(word2idx)} 词  平均长度 {np.mean(lengths):.1f}  "
          f"最长 {max(lengths)}  截断到 {args.max_len}")

    emb = load_glove(word2idx, args.dim)

    results = {}
    for v in args.variant:
        print(f"\n=== {VARIANT_NAMES[v]} ===")
        accs = run_variant(v, texts, labels, word2idx, emb, args, device)
        mean, std = float(np.mean(accs) * 100), float(np.std(accs) * 100)
        results[v] = {"name": VARIANT_NAMES[v], "folds": [round(a * 100, 2) for a in accs],
                      "mean": round(mean, 2), "std": round(std, 2), "paper": PAPER_MR[v]}
        print(f"  → {VARIANT_NAMES[v]}: {mean:.2f} ± {std:.2f}   (论文 {PAPER_MR[v]})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "device": device, "results": results}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== 汇总（MR，10 折交叉验证准确率 %）===")
    print(f"{'模型':<18}{'本次复现':>16}{'论文报告':>12}{'差值':>10}")
    for v in args.variant:
        r = results[v]
        print(f"{r['name']:<18}{r['mean']:>10.2f} ± {r['std']:<4.2f}{r['paper']:>12.1f}"
              f"{r['mean']-r['paper']:>+10.2f}")
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
