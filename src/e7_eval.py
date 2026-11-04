"""E7 result evaluation on the E6 gap set, restricted to held-out questions.

Same gap definition as E6 (dense rank in (3,50]), same top-20 candidates, but
only questions in the 30% test split of the distillation (group split by qid).
Compares base ms-marco, the distilled student, and the LLM teacher.
"""
import argparse
import json
import os
import random

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


def test_qids():
    recs = [json.loads(l) for l in open(os.path.join(RES, "e7_pairs.jsonl"), encoding="utf-8") if l.strip()]
    qids = sorted({r["qid"] for r in recs})
    random.Random(0).shuffle(qids)
    n = int(len(qids) * 0.7)
    return set(qids[n:]), set(qids[:n])


def score(mdl, tok, pairs, max_len=128, batch=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            c = pairs[i:i + batch]
            b = tok([x[0] for x in c], [x[1] for x in c], truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt")
            out.extend(mdl(**b).logits.view(-1).tolist())
    return out


def main(n=200, topn=20):
    te, tr = test_qids()
    print(f"[e7-eval] held-out qids={len(te)}  train qids={len(tr)}")

    qs = load()[:n]
    ce = CERemote()
    rows, corpus, meta, emb = measure(qs, ce)
    import src.e5_gap as M
    M._GOLD = build(qs)[2]
    gold = M._GOLD

    gaps = [r for r in rows if r["rank_dense"] and 3 < r["rank_dense"] <= 50]
    gaps_te = [r for r in gaps if r["qid"] in te]
    print(f"[e7-eval] gap cases total={len(gaps)}  held-out={len(gaps_te)}")

    tok = AutoTokenizer.from_pretrained(DIST)
    dist = AutoModelForSequenceClassification.from_pretrained(DIST).eval()
    tok0 = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSequenceClassification.from_pretrained(BASE).eval()

    def evaluate(items, label):
        groups = {}
        for r in items:
            top = list(np.argsort(-r["dense_scores"])[:topn])
            pairs = [(r["text"], corpus[i][:1200]) for i in top]
            groups[(r["qid"], r["hop"])] = {
                "top": top, "gold": set(gold.get((r["qid"], r["hop"]), [])),
                "base": score(base, tok0, pairs), "dist": score(dist, tok, pairs),
            }
        for name in ["base", "dist"]:
            reach = hit = 0
            for g in groups.values():
                if not (g["gold"] & set(g["top"])):
                    continue
                reach += 1
                order = sorted(range(len(g["top"])), key=lambda j: -g[name][j])
                hit += any(g["top"][j] in g["gold"] for j in order[:3])
            print(f"  [{label}] {name}: reachable={reach} closure@3={hit/reach:.3f}")
        # LLM teacher from E6 details (same qid/hop, same top-20 order)
        e6 = json.load(open(os.path.join(RES, "e6_llm_upper.json")))
        llm = {(d["qid"], d["hop"]): d for d in e6["details"]}
        reach = hit = 0
        for k, g in groups.items():
            d = llm.get(k)
            if d is None or not (g["gold"] & set(g["top"])):
                continue
            reach += 1
            order = sorted(range(len(g["top"])), key=lambda j: (-d["llm_scores"][j], -g["base"][j]))
            hit += any(g["top"][j] in g["gold"] for j in order[:3])
        print(f"  [{label}] llm: reachable={reach} closure@3={hit/reach if reach else float('nan'):.3f}")
        return groups

    print("held-out questions:")
    gte = evaluate(gaps_te, "held-out")
    print("all questions (leaky, reference only):")
    evaluate(gaps, "all")

    # per-case detail for held-out
    out = []
    for (qid, hop), g in gte.items():
        order_d = sorted(range(len(g["top"])), key=lambda j: -g["dist"][j])
        order_b = sorted(range(len(g["top"])), key=lambda j: -g["base"][j])
        out.append({"qid": qid, "hop": hop,
                    "base_rank": next((p + 1 for p, j in enumerate(order_b) if g["top"][j] in g["gold"]), None),
                    "dist_rank": next((p + 1 for p, j in enumerate(order_d) if g["top"][j] in g["gold"]), None)})
    with open(os.path.join(RES, "e7_eval_heldout.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--topn", type=int, default=20)
    a = ap.parse_args()
    main(a.n, a.topn)
