"""E11: pointwise distillation + pairwise preference margin.

Loss = BCE(pointwise teacher score) + lambda * margin(gold >= neg + m).
Pairs with a TIE were dropped upstream.
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


def load_cfg():
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class PointDS(Dataset):
    def __init__(self, recs, tok, ml):
        self.r, self.t, self.ml = recs, tok, ml
    def __len__(self):
        return len(self.r)
    def __getitem__(self, i):
        r = self.r[i]
        b = self.t(r["q"], r["p"], truncation=True, max_length=self.ml,
                   padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in b.items()}, torch.tensor(r["score"] / 10.0)


class PairDS(Dataset):
    def __init__(self, recs, tok, ml):
        self.r, self.t, self.ml = recs, tok, ml
    def __len__(self):
        return len(self.r)
    def __getitem__(self, i):
        r = self.r[i]
        a = self.t(r["q"], r["text_a"], truncation=True, max_length=self.ml,
                   padding="max_length", return_tensors="pt")
        b = self.t(r["q"], r["text_b"], truncation=True, max_length=self.ml,
                   padding="max_length", return_tensors="pt")
        return ({k: v.squeeze(0) for k, v in a.items()},
                {k: v.squeeze(0) for k, v in b.items()},
                torch.tensor(float(r["pref"])))


def infer(mdl, tok, pairs, dev, ml, bs=128):
    mdl.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), bs):
            c = pairs[i:i + bs]
            b = tok([x[0] for x in c], [x[1] for x in c], truncation=True,
                    max_length=ml, padding=True, return_tensors="pt").to(dev)
            out.extend(mdl(**b).logits.view(-1).float().cpu().tolist())
    return out


def evaluate(te, base_s, dist_s):
    groups = {}
    for i, r in enumerate(te):
        groups.setdefault((r["qid"], r["hop"]), []).append((r, i))
    res = {}
    for name, s in [("base", base_s), ("distilled", dist_s)]:
        bucket = {"all": [0, 0], "easy": [0, 0], "gap": [0, 0], "conf_wrong": [0, 0]}
        for k, items in groups.items():
            if not any(r["label"] == 1 for r, _ in items):
                continue
            order = sorted(items, key=lambda x: -s[x[1]])
            ok = any(r["label"] == 1 for r, _ in order[:3])
            gr = items[0][0]["gold_rank"]
            for b in ["all", "easy" if gr <= 3 else "gap"]:
                bucket[b][0] += 1
                bucket[b][1] += int(ok)
            if any(sig(r["base_logit"]) < 0.2 and r["label"] == 1 for r, _ in items):
                bucket["conf_wrong"][0] += 1
                bucket["conf_wrong"][1] += int(ok)
        res[name] = {k: {"n": v[0], "closure@3": v[1] / v[0] if v[0] else None} for k, v in bucket.items()}
    return res


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointwise", default=os.path.join(HERE, "pairs_pointwise.jsonl"))
    ap.add_argument("--pairwise", default=os.path.join(HERE, "pairs_pairwise.jsonl"))
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--lr", type=float, default=cfg["lr"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    pt = read(a.pointwise)
    pr = read(a.pairwise)
    qids = sorted({r["qid"] for r in pt})
    random.Random(cfg["seed"]).shuffle(qids)
    n = int(len(qids) * cfg["train_frac"])
    tr_q, te_q = set(qids[:n]), set(qids[n:])
    tr_pt = [r for r in pt if r["qid"] in tr_q]
    te = [r for r in pt if r["qid"] in te_q]
    tr_pr = [r for r in pr if r["qid"] in tr_q]

    def hard(r):
        p = sig(r["base_logit"])
        return (r["score"] >= 8 and p < 0.2) or (r["score"] < 8 and p > 0.8)
    hard_tr = [r for r in tr_pt if hard(r)]
    tr_pt = tr_pt + hard_tr * (cfg["oversample_confident_wrong"] - 1)
    if a.smoke:
        tr_pt, tr_pr, te, a.epochs = tr_pt[:128], tr_pr[:64], te[:128], 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} pointwise={len(tr_pt)} pairwise={len(tr_pr)} test={len(te)}")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg["model"]).to(dev)
    opt = torch.optim.AdamW(mdl.parameters(), lr=a.lr)
    dl_p = DataLoader(PointDS(tr_pt, tok, cfg["max_length"]), batch_size=cfg["pointwise_batch"], shuffle=True)
    dl_q = DataLoader(PairDS(tr_pr, tok, cfg["max_length"]), batch_size=cfg["pairwise_batch"], shuffle=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    it_q = iter(dl_q)

    t0 = time.time()
    mdl.train()
    for ep in range(a.epochs):
        tot = cnt = 0
        for b, y in dl_p:
            try:
                ba, bb, pref = next(it_q)
            except StopIteration:
                it_q = iter(dl_q)
                ba, bb, pref = next(it_q)
            b = {k: v.to(dev) for k, v in b.items()}
            ba = {k: v.to(dev) for k, v in ba.items()}
            bb = {k: v.to(dev) for k, v in bb.items()}
            y, pref = y.to(dev), pref.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                lp = mdl(**b).logits.view(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lp, y)
                sa = mdl(**ba).logits.view(-1)
                sb = mdl(**bb).logits.view(-1)
                signed = torch.where(pref > 0.5, sa - sb, sb - sa)
                loss = loss + cfg["lambda_pair"] * torch.nn.functional.relu(cfg["margin"] - signed).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); cnt += 1
            if cnt % 100 == 0:
                print(f"  ep{ep+1} {cnt}/{len(dl_p)} loss={tot/cnt:.4f}")
    print(f"train {time.time()-t0:.1f}s")

    outdir = os.path.join(HERE, "model")
    os.makedirs(outdir, exist_ok=True)
    mdl.save_pretrained(outdir); tok.save_pretrained(outdir)

    base = AutoModelForSequenceClassification.from_pretrained(cfg["model"]).to(dev)
    pairs = [(r["q"], r["p"]) for r in te]
    res = evaluate(te, infer(base, tok, pairs, dev, cfg["max_length"]), infer(mdl, tok, pairs, dev, cfg["max_length"]))
    res["train_pointwise"] = len(tr_pt); res["train_pairwise"] = len(tr_pr); res["test"] = len(te)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
