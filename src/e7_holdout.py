"""E7 held-out evaluation on questions the distilled model never saw.

MuSiQue questions 200-400, chunk-level corpus, same gap definition as E5/E6.
Scores: base ms-marco vs distilled, closure@3 over reachable gap cases.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .ce_client import CERemote
from .e5_gap import build, measure
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DIST = os.path.join(ROOT, "models", "e7_distilled", "model")


def score(mdl, tok, pairs, max_len=128, batch=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            c = pairs[i:i + batch]
            b = tok([x[0] for x in c], [x[1] for x in c], truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt")
            out.extend(mdl(**b).logits.view(-1).tolist())
    return out


def main(start=200, end=400, topn=20):
    qs = load(os.path.join(ROOT, "data", "raw", "musique400.jsonl"))[start:end]
    print(f"[e7-holdout] questions {start}:{end} -> {len(qs)}")
    ce = CERemote()
    rows, corpus, meta, emb = measure(qs, ce)
    import src.e5_gap as M
    M._GOLD = build(qs)[2]
    gold = M._GOLD

    gaps = [r for r in rows if r["rank_dense"] and 3 < r["rank_dense"] <= 50]
    print(f"[e7-holdout] chunks={len(corpus)} hops={len(rows)} gap cases={len(gaps)}")

    tok = AutoTokenizer.from_pretrained(DIST)
    dist = AutoModelForSequenceClassification.from_pretrained(DIST).eval()
    tok0 = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSequenceClassification.from_pretrained(BASE).eval()

    res = {"base": [], "dist": []}
    detail = []
    for r in gaps:
        top = list(np.argsort(-r["dense_scores"])[:topn])
        ids = set(gold.get((r["qid"], r["hop"]), []))
        pairs = [(r["text"], corpus[i][:1200]) for i in top]
        sb = score(base, tok0, pairs)
        sd = score(dist, tok, pairs)
        reachable = bool(ids & set(top))
        for name, s in [("base", sb), ("dist", sd)]:
            order = sorted(range(len(top)), key=lambda j: -s[j])
            hit = reachable and any(top[j] in ids for j in order[:3])
            res[name].append((reachable, hit))
        detail.append({"qid": r["qid"], "hop": r["hop"], "dense_rank": r["rank_dense"],
                       "reachable": reachable,
                       "base_rank": next((p + 1 for p, j in enumerate(sorted(range(len(top)), key=lambda j: -sb[j])) if top[j] in ids), None),
                       "dist_rank": next((p + 1 for p, j in enumerate(sorted(range(len(top)), key=lambda j: -sd[j])) if top[j] in ids), None)})

    out = {"n_questions": len(qs), "n_chunks": len(corpus), "n_hops": len(rows),
           "gap_cases": len(gaps)}
    for name in ["base", "dist"]:
        reach = [h for r, h in res[name] if r]
        out[name] = {"reachable": len(reach), "closure@3": sum(reach) / len(reach) if reach else None}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(os.path.join(RES, "e7_holdout.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "details": detail}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--end", type=int, default=400)
    ap.add_argument("--topn", type=int, default=20)
    a = ap.parse_args()
    main(a.start, a.end, a.topn)
