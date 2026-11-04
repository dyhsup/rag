"""Distill the LLM gap judge into a cross-encoder. Self-contained.

Input : pairs.jsonl  (qid, hop, idx, q, p, score 0-10, is_gold, dense_rank)
Output: ./model/     (trained cross-encoder)  +  metrics.json

Usage:
    python train_gpu.py                     # config.json defaults
    python train_gpu.py --epochs 3 --batch 64
    python train_gpu.py --smoke             # 64 pairs, 1 epoch, sanity check
"""
import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    path = path or os.path.join(HERE, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_split(recs, frac, seed):
    qids = sorted({r["qid"] for r in recs})
    random.Random(seed).shuffle(qids)
    n = int(len(qids) * frac)
    tr, te = set(qids[:n]), set(qids[n:])
    return ([r for r in recs if r["qid"] in tr], [r for r in recs if r["qid"] in te])


class PairDS(Dataset):
    def __init__(self, recs, tok, max_len):
        self.recs, self.tok, self.max_len = recs, tok, max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        b = self.tok(r["q"], r["p"], truncation=True, max_length=self.max_len,
                     padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in b.items()}, torch.tensor(
            r["score"] / 10.0, dtype=torch.float)


def predict(model, tok, pairs, device, batch=128, max_len=128):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            b = tok([x[0] for x in chunk], [x[1] for x in chunk], truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt").to(device)
            out.extend(model(**b).logits.view(-1).float().cpu().tolist())
    return out


def evaluate(te, base_scores, dist_scores):
    groups = {}
    for i, r in enumerate(te):
        groups.setdefault((r["qid"], r["hop"]), []).append((r, i))
    res = {}
    for name, scores in [("base_msmarco", base_scores), ("distilled", dist_scores)]:
        reach = hit = gap_n = gap_hit = 0
        for key, items in groups.items():
            if not any(r["is_gold"] for r, _ in items):
                continue
            reach += 1
            order = sorted(items, key=lambda x: -scores[x[1]])
            ok = any(r["is_gold"] for r, _ in order[:3])
            hit += ok
            dr = items[0][0].get("dense_rank")
            if dr and dr > 3:
                gap_n += 1
                gap_hit += ok
        res[name] = {
            "reachable_cases": reach,
            "closure@3": hit / reach if reach else None,
            "gap_cases": gap_n,
            "gap_closure@3": gap_hit / gap_n if gap_n else None,
        }
    return res


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(HERE, "pairs.jsonl"))
    ap.add_argument("--model", default=cfg["model"])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--lr", type=float, default=cfg["lr"])
    ap.add_argument("--batch", type=int, default=cfg["batch_size"])
    ap.add_argument("--max-len", type=int, default=cfg["max_length"])
    ap.add_argument("--train-frac", type=float, default=cfg["train_frac"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--out", default=os.path.join(HERE, "model"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    recs = load_pairs(a.pairs)
    tr, te = group_split(recs, a.train_frac, a.seed)
    if a.smoke:
        tr, te = tr[:64], te[:64]
        a.epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} pairs={len(recs)} train={len(tr)} test={len(te)}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    dl = DataLoader(PairDS(tr, tok, a.max_len), batch_size=a.batch, shuffle=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    model.train()
    t0 = time.time()
    for ep in range(a.epochs):
        total = 0.0
        for i, (b, y) in enumerate(dl, 1):
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(**b).logits.view(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            if i % 20 == 0:
                print(f"  ep{ep+1} {i}/{len(dl)} loss={total/i:.4f}")
    print(f"train time {time.time()-t0:.1f}s")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)

    pairs = [(r["q"], r["p"]) for r in te]
    base = AutoModelForSequenceClassification.from_pretrained(a.model).to(device)
    base_scores = predict(base, tok, pairs, device, max_len=a.max_len)
    dist_scores = predict(model, tok, pairs, device, max_len=a.max_len)
    res = evaluate(te, base_scores, dist_scores)
    res["train_size"] = len(tr)
    res["test_size"] = len(te)
    res["device"] = device
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
