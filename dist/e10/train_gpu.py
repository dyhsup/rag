"""E8: distill the LLM gap judge with hard-example oversampling.

pairs.jsonl fields: qid, hop, idx, q, p, kind, label, dense_rank, gold_rank,
                    score (teacher 0-10), base_logit (ms-marco before distillation)

Key change vs E7: the 892 pairs where the base CE is confidently wrong are
oversampled, because that is where distillation can actually add value.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def sig(x):
    return 1 / (1 + math.exp(-x))


def load_config(p=None):
    with open(p or os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        return json.load(f)


class DS(Dataset):
    def __init__(self, recs, tok, max_len):
        self.recs, self.tok, self.max_len = recs, tok, max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        b = self.tok(r["q"], r["p"], truncation=True, max_length=self.max_len,
                     padding="max_length", return_tensors="pt")
        return ({k: v.squeeze(0) for k, v in b.items()},
                torch.tensor(r["score"] / 10.0, dtype=torch.float))


def predict(model, tok, pairs, device, max_len=128, batch=128):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            c = pairs[i:i + batch]
            b = tok([x[0] for x in c], [x[1] for x in c], truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt").to(device)
            out.extend(model(**b).logits.view(-1).float().cpu().tolist())
    return out


def evaluate(te, base_s, dist_s):
    groups = {}
    for i, r in enumerate(te):
        groups.setdefault((r["qid"], r["hop"]), []).append((r, i))
    res = {}
    for name, s in [("base_msmarco", base_s), ("distilled", dist_s)]:
        bucket = {"all": [0, 0], "easy(gold<=3)": [0, 0], "gap(gold>3)": [0, 0],
                  "base_conf_wrong": [0, 0]}
        for key, items in groups.items():
            if not any(r["label"] == 1 for r, _ in items):
                continue
            order = sorted(items, key=lambda x: -s[x[1]])
            ok = any(r["label"] == 1 for r, _ in order[:3])
            gr = items[0][0]["gold_rank"]
            for b in ["all", "easy(gold<=3)" if gr <= 3 else "gap(gold>3)"]:
                bucket[b][0] += 1
                bucket[b][1] += int(ok)
            if any(sig(r["base_logit"]) < 0.2 and r["label"] == 1 for r, _ in items):
                bucket["base_conf_wrong"][0] += 1
                bucket["base_conf_wrong"][1] += int(ok)
        res[name] = {k: {"n": v[0], "closure@3": v[1] / v[0] if v[0] else None}
                     for k, v in bucket.items()}
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
    ap.add_argument("--oversample", type=int, default=cfg["oversample_confident_wrong"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]
    qids = sorted({r["qid"] for r in recs})
    random.Random(a.seed).shuffle(qids)
    n = int(len(qids) * a.train_frac)
    tr_q, te_q = set(qids[:n]), set(qids[n:])
    tr = [r for r in recs if r["qid"] in tr_q]
    te = [r for r in recs if r["qid"] in te_q]

    # oversample pairs where the base CE is confidently wrong
    def hard(r):
        p = sig(r["base_logit"])
        return (r["score"] >= 8 and p < 0.2) or (r["score"] < 8 and p > 0.8)

    hard_tr = [r for r in tr if hard(r)]
    tr_aug = tr + hard_tr * max(0, a.oversample - 1)
    if a.smoke:
        tr_aug, te, a.epochs = tr_aug[:128], te[:128], 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} train={len(tr)} hard={len(hard_tr)} augmented={len(tr_aug)} test={len(te)}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    dl = DataLoader(DS(tr_aug, tok, a.max_len), batch_size=a.batch, shuffle=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        tot = 0.0
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
            tot += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep+1} {i}/{len(dl)} loss={tot/i:.4f}")
    print(f"train {time.time()-t0:.1f}s")

    out = os.path.join(HERE, "model")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    base = AutoModelForSequenceClassification.from_pretrained(a.model).to(device)
    pairs = [(r["q"], r["p"]) for r in te]
    res = evaluate(te, predict(base, tok, pairs, device, a.max_len),
                   predict(model, tok, pairs, device, a.max_len))
    res["train_size"] = len(tr)
    res["hard_train"] = len(hard_tr)
    res["test_size"] = len(te)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
