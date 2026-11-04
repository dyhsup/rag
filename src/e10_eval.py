"""Evaluate the E10 distilled matcher on the E5/E6 gap protocol.

Held-out questions = the 30% qid split used by the E10 training (seed 0, 0.7).
Candidates = dense top-20; gap case = dense rank in (3,50]; gold = chunk that
contains the hop answer. Report base vs distilled closure@3 with a paired CI.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import eval as ev
from .e5_gap import build
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DIST = os.path.join(ROOT, "models", "e10_distilled", "model")


def split_qids():
    recs = [json.loads(l) for l in open(os.path.join(RES, "e10_scored.jsonl"), encoding="utf-8") if l.strip()]
    qids = sorted({r["qid"] for r in recs})
    random.Random(0).shuffle(qids)
    n = int(len(qids) * 0.7)
    return set(qids[n:]), set(qids[:n])


def score(mdl, tok, pairs, batch=64, max_len=128):
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            c = pairs[i:i + batch]
            b = tok([x[0] for x in c], [x[1] for x in c], truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt")
            out.extend(mdl(**b).logits.view(-1).tolist())
    return out


def main(n=2417, topn=20):
    te, tr = split_qids()
    print(f"[e10-eval] held-out qids={len(te)} train qids={len(tr)}")

    qs_all = load(os.path.join(ROOT, "data", "raw", "musique2417.jsonl"))[:n]
    corpus, meta, gold, _ = build(qs_all)
    qs = [q for q in qs_all if q["qid"] in te]
    print(f"[e10-eval] held-out questions={len(qs)} chunks={len(corpus)} hops={len(gold)}")

    D = np.load(os.path.join(IDX, f"e8_chunks_175642.npy"))
    assert D.shape[0] == len(corpus), (D.shape, len(corpus))

    tok = AutoTokenizer.from_pretrained(DIST)
    dist = AutoModelForSequenceClassification.from_pretrained(DIST).eval()
    tok0 = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSequenceClassification.from_pretrained(BASE).eval()

    hits = {"base": [0, 0], "dist": [0, 0]}
    diffs = []
    gaps = 0
    # batch-encode all hop queries
    hop_rows = [(q, h) for q in qs for h in q["hops"] if gold.get((q["qid"], h["hop"]))]
    from .ce_client import CERemote
    from .e2b_ce import render_hop
    ce = CERemote()
    texts = [render_hop(q, h["hop"]) for q, h in hop_rows]
    Q = ce.embed(texts)
    for i, (q, h) in enumerate(hop_rows):
        ids = set(gold[(q["qid"], h["hop"])])
        s = D @ Q[i]
        order = np.argsort(-s)
        pos = {int(idx): r + 1 for r, idx in enumerate(order) if int(idx) in ids}
        if not pos:
            continue
        gr = min(pos.values())
        if not (3 < gr <= 50):
            continue
        gaps += 1
        top = [int(x) for x in order[:topn]]
        pairs = [(texts[i], corpus[j][:1200]) for j in top]
        sb = score(base, tok0, pairs)
        sd = score(dist, tok, pairs)
        reach = bool(ids & set(top))
        for name, sc in [("base", sb), ("dist", sd)]:
            order2 = sorted(range(len(top)), key=lambda j: -sc[j])
            ok = reach and any(top[j] in ids for j in order2[:3])
            hits[name][1] += int(reach)
            hits[name][0] += int(ok)
            if name == "dist":
                base_order = sorted(range(len(top)), key=lambda j: -sb[j])
                base_ok = reach and any(top[j] in ids for j in base_order[:3])
                diffs.append(int(ok) - int(base_ok))

    out = {"n_questions": len(qs), "gap_cases": gaps,
           "base": {"reachable": hits["base"][1], "closure@3": hits["base"][0] / max(1, hits["base"][1])},
           "distilled": {"reachable": hits["dist"][1], "closure@3": hits["dist"][0] / max(1, hits["dist"][1])},
           "paired_delta": ev.paired_bootstrap(diffs, diffs)}
    out["paired_delta"] = {"mean": ev.mean(diffs), "ci": ev.bootstrap_ci(diffs)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(os.path.join(RES, "e10_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2417)
    ap.add_argument("--topn", type=int, default=20)
    a = ap.parse_args()
    main(a.n, a.topn)
