"""Unified comparison on the held-out set and the full 175k-chunk corpus.

Same top-20 dense candidates for every scorer:
  dense pointwise | BM25 | base ms-marco CE | E10 distilled | LLM judge
Metrics: closure@3 and closure@10 over reachable gap cases (dense rank 3-50).
"""
import argparse
import json
import math
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import eval as ev
from .ce_client import CERemote
from .e2b_ce import render_hop
from .e5_gap import build
from .e6_llm_upper import JUDGE, parse_int
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DIST = os.path.join(ROOT, "models", "e10_distilled", "model")
_lock = threading.Lock()


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def build_idf(corpus, k1=1.5, b=0.75):
    df = Counter()
    total = 0
    for c in corpus:
        ts = tok(c)
        total += len(ts)
        df.update(set(ts))
    n = len(corpus)
    avgdl = total / n
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    return idf, avgdl, k1, b


def bm25(q, doc, idf, avgdl, k1, b):
    ts = tok(doc)
    dl = len(ts)
    tf = Counter(ts)
    s = 0.0
    for t, f in tf.items():
        if t not in idf:
            continue
        s += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
    return s


def score_ce(mdl, tokenizer, pairs, batch=64, max_len=128):
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            c = pairs[i:i + batch]
            b = tokenizer([x[0] for x in c], [x[1] for x in c], truncation=True,
                          max_length=max_len, padding=True, return_tensors="pt")
            out.extend(mdl(**b).logits.view(-1).tolist())
    return out


def split_qids():
    recs = [json.loads(l) for l in open(os.path.join(RES, "e10_scored.jsonl"), encoding="utf-8") if l.strip()]
    qids = sorted({r["qid"] for r in recs})
    import random
    random.Random(0).shuffle(qids)
    n = int(len(qids) * 0.7)
    return set(qids[n:])


def main(n=2417, topn=20, workers=12):
    te = split_qids()
    qs_all = load(os.path.join(ROOT, "data", "raw", "musique2417.jsonl"))[:n]
    corpus, meta, gold, _ = build(qs_all)
    D = np.load(os.path.join(IDX, "e8_chunks_175642.npy"))
    print(f"[cmp] held-out qids={len(te)} corpus={len(corpus)}")

    hop_rows = [{"qid": q["qid"], "hop": h["hop"], "text": render_hop(q, h["hop"])}
                for q in qs_all if q["qid"] in te
                for h in q["hops"] if gold.get((q["qid"], h["hop"]))]
    ce = CERemote()
    Q = ce.embed([x["text"] for x in hop_rows])
    print(f"[cmp] held-out hops={len(hop_rows)}")

    cases = []
    for i, hr in enumerate(hop_rows):
        ids = set(gold[(hr["qid"], hr["hop"])])
        s = D @ Q[i]
        order = np.argsort(-s)
        pos = {}
        for r, idx in enumerate(order):
            if int(idx) in ids:
                pos[int(idx)] = r + 1
                if len(pos) >= 3:
                    break
        if not pos:
            continue
        gr = min(pos.values())
        if not (3 < gr <= 50):
            continue
        top = [int(x) for x in order[:topn]]
        if not (ids & set(top)):
            continue
        cases.append({"qid": hr["qid"], "hop": hr["hop"], "text": hr["text"],
                      "top": top, "ids": ids, "dense": [float(s[j]) for j in top]})
    print(f"[cmp] reachable gap cases={len(cases)}")

    idf, avgdl, k1, b = build_idf(corpus)
    print(f"[cmp] idf terms={len(idf)} avgdl={avgdl:.1f}")

    tok0 = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSequenceClassification.from_pretrained(BASE).eval()
    tokd = AutoTokenizer.from_pretrained(DIST)
    dist = AutoModelForSequenceClassification.from_pretrained(DIST).eval()

    # LLM scoring
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    jobs = [(ci, j, corpus[idx][:1200]) for ci, c in enumerate(cases) for j, idx in enumerate(c["top"])]
    print(f"[cmp] LLM pairs={len(jobs)}")
    llm_scores = [[0] * topn for _ in cases]

    def work(job):
        ci, j, p = job
        try:
            return ci, j, parse_int(llm.chat(
                [{"role": "user", "content": JUDGE.format(q=cases[ci]["text"], p=p)}], max_tokens=8))
        except Exception:
            return ci, j, 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            ci, j, sc = fut.result()
            llm_scores[ci][j] = sc
            if k % 1000 == 0 or k == len(jobs):
                print(f"[cmp] llm {k}/{len(jobs)} calls={llm.calls}", flush=True)

    names = ["dense", "bm25", "base_ce", "distilled", "llm"]
    hits3 = {x: [] for x in names}
    hits10 = {x: [] for x in names}
    for ci, c in enumerate(cases):
        texts = [corpus[i] for i in c["top"]]
        sb = score_ce(base, tok0, [(c["text"], t[:1200]) for t in texts])
        sd = score_ce(dist, tokd, [(c["text"], t[:1200]) for t in texts])
        sc = {"dense": c["dense"], "bm25": [bm25(c["text"], t, idf, avgdl, k1, b) for t in texts],
              "base_ce": sb, "distilled": sd, "llm": llm_scores[ci]}
        for name in names:
            order = sorted(range(len(texts)), key=lambda j: -sc[name][j])
            hits3[name].append(int(any(c["top"][j] in c["ids"] for j in order[:3])))
            hits10[name].append(int(any(c["top"][j] in c["ids"] for j in order[:10])))

    out = {"n_questions": len(te), "n_hops": len(hop_rows), "reachable_gap_cases": len(cases),
           "scores": {}}
    for name in names:
        out["scores"][name] = {
            "closure@3": {"mean": ev.mean(hits3[name]), "ci": ev.bootstrap_ci(hits3[name])},
            "closure@10": {"mean": ev.mean(hits10[name]), "ci": ev.bootstrap_ci(hits10[name])},
        }
    for name in ["bm25", "base_ce", "distilled", "llm"]:
        out["scores"][name]["delta@3_vs_base"] = ev.paired_bootstrap(hits3[name], hits3["base_ce"])
        out["scores"][name]["delta@3_vs_dense"] = ev.paired_bootstrap(hits3[name], hits3["dense"])

    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(os.path.join(RES, "e10_compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2417)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    main(a.n, a.topn, a.workers)
