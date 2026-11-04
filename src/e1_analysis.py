"""Characterize the hops that first-stage retrieval misses.

For every hop, find the rank of its gold paragraph under BM25 and dense, and
compare lexical overlap for hits vs misses. If misses are lexically disjoint but
semantically valid, the failure is the query-evidence gap itself.
"""
import json
import os

import numpy as np

from .data import tokenize
from .musique import load
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
TAG = "all-MiniLM-L6-v2"


def jaccard(a, b):
    A, B = set(tokenize(a)), set(tokenize(b))
    return len(A & B) / max(1, len(A | B))


def rank_of_row(scores, idx):
    return int((scores > scores[idx]).sum()) + 1


def run(n=200):
    qs = load()[:n]
    docs, keys, owner = [], [], []
    for q in qs:
        for d in q["docs"]:
            docs.append({"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
                         "sentences": d["sentences"]})
            keys.append((d["title"], d["text"]))
    idx = QuestionIndex({"qid": "pool", "docs": docs})

    D = np.load(os.path.join(IDX, f"doc_{TAG}.npy"))
    Q = np.load(os.path.join(IDX, f"q_{TAG}.npy"))[:n]
    S = Q @ D.T

    hops = []
    for qi, q in enumerate(qs):
        bm = np.array(idx.score_docs(q["question"]))
        dn = S[qi]
        for h in q["hops"]:
            gold_doc = next((d for d in q["docs"] if d["idx"] == h["para_idx"]), None)
            if gold_doc is None:
                continue
            gkey = (gold_doc["title"], gold_doc["text"])
            try:
                gi = keys.index(gkey)
            except ValueError:
                continue
            hops.append({
                "qid": q["qid"], "hop": h["hop"], "n_hops": q["n_hops"],
                "question": q["question"], "hop_text": h["question"],
                "rank_bm25": rank_of_row(bm, gi), "rank_dense": rank_of_row(dn, gi),
                "jac_q": jaccard(q["question"], gold_doc["text"]),
                "jac_hop": jaccard(h["question"], gold_doc["text"]),
                "gold_title": gold_doc["title"],
            })

    def bucket(r):
        if r == 1: return "1"
        if r <= 5: return "2-5"
        if r <= 50: return "6-50"
        if r <= 100: return "51-100"
        return ">100"

    out = {"n_hops": len(hops), "hops": hops}
    for name in ["rank_dense", "rank_bm25"]:
        groups = {}
        for h in hops:
            groups.setdefault(bucket(h[name]), []).append(h)
        out[name] = {
            b: {"n": len(v), "jac_q": sum(x["jac_q"] for x in v) / len(v),
                "jac_hop": sum(x["jac_hop"] for x in v) / len(v)}
            for b, v in sorted(groups.items(), key=lambda kv: {"1": 0, "2-5": 1, "6-50": 2, "51-100": 3, ">100": 4}[kv[0]])
        }

    with open(os.path.join(RES, "e1_gap_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    report(out)


def report(out):
    lines = ["# First-stage failure analysis (MuSiQue, 200 questions)", ""]
    lines.append(f"- hops: {out['n_hops']}")
    lines.append("")
    for name, label in [("rank_dense", "dense (MiniLM)"), ("rank_bm25", "BM25")]:
        lines.append(f"## gold rank under {label}")
        lines.append("")
        lines.append("| rank bucket | hops | Jaccard(question, gold) | Jaccard(hop, gold) |")
        lines.append("|---|---|---|---|")
        for b, d in out[name].items():
            lines.append(f"| {b} | {d['n']} | {d['jac_q']:.3f} | {d['jac_hop']:.3f} |")
        lines.append("")
    miss = [h for h in out["hops"] if h["rank_dense"] > 100]
    lines.append(f"## examples of dense misses (rank > 100): {len(miss)}")
    lines.append("")
    for h in miss[:8]:
        lines.append(f"- {h['question'][:80]} | gold={h['gold_title']} | rank={h['rank_dense']} | jac_q={h['jac_q']:.2f}")
    path = os.path.join(RES, "e1_gap_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[analysis] wrote {path}")


if __name__ == "__main__":
    run()
