"""Per-dataset CE fine-tuning.

Usage:
    python train_ce.py --data musique.jsonl --out models/musique
    python train_ce.py --data trivia.jsonl --out models/trivia --epochs 3

Group split by qid for the reported AUC; the saved model is trained on the
train split (set --val-frac 0 to use all pairs).
"""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class DS(Dataset):
    def __init__(self, recs, tok, ml):
        self.r, self.t, self.ml = recs, tok, ml

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        r = self.r[i]
        b = self.t(r["q"], r["p"], truncation=True, max_length=self.ml,
                   padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in b.items()}, torch.tensor(float(r["label"]))


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def predict(model, tok, recs, dev, ml, bs=64):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(recs), bs):
            c = recs[i:i + bs]
            b = tok([r["q"] for r in c], [r["p"] for r in c], truncation=True,
                    max_length=ml, padding=True, return_tensors="pt").to(dev)
            out.extend(model(**b).logits.view(-1).float().cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    recs = read(a.data)
    qids = sorted({r["qid"] for r in recs})
    random.Random(a.seed).shuffle(qids)
    n_val = int(len(qids) * a.val_frac)
    val_q = set(qids[:n_val])
    tr = [r for r in recs if r["qid"] not in val_q] if n_val else recs
    va = [r for r in recs if r["qid"] in val_q]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} train={len(tr)} val={len(va)} qids={len(qids)}")
    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev)
    opt = torch.optim.AdamW(mdl.parameters(), lr=a.lr)
    dl = DataLoader(DS(tr, tok, a.max_len), batch_size=a.batch, shuffle=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    if va:
        base = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev)
        s0 = predict(base, tok, va, dev, a.max_len)
        print(f"zero-shot val AUC={auc(s0, [r['label'] for r in va]):.3f}")

    mdl.train()
    for ep in range(a.epochs):
        tot = 0
        for i, (b, y) in enumerate(dl, 1):
            b = {k: v.to(dev) for k, v in b.items()}
            y = y.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                logits = mdl(**b).logits.view(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep+1} {i}/{len(dl)} loss={tot/i:.4f}")

    os.makedirs(a.out, exist_ok=True)
    mdl.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    if va:
        s1 = predict(mdl, tok, va, dev, a.max_len)
        print(f"fine-tuned val AUC={auc(s1, [r['label'] for r in va]):.3f}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
