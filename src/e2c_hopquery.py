"""E2c: is the later-hop recall collapse caused by the missing binding?

Retrieve with the *bound* requirement query (previous hop's answer substituted
into the hop) instead of the whole question. Oracle binding, so this measures
the ceiling a plan-driven controller could reach if hop h-1 is answered right.
"""
import json
import os

import numpy as np

from .e1_analysis import rank_of_row
from .e1_dense import encode
from .e2b_ce import render_hop
from .musique import load
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
TAG = "all-MiniLM-L6-v2"


def run(n=200):
    qs = load()[:n]
    docs, keys = [], []
    for q in qs:
        for d in q["docs"]:
            docs.append({"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
                         "sentences": d["sentences"]})
            keys.append((d["title"], d["text"]))
    idx = QuestionIndex({"qid": "pool", "docs": docs})
    D = np.load(os.path.join(IDX, f"doc_{TAG}.npy"))

    items, texts = [], []
    for q in qs:
        for h in q["hops"]:
            texts.append(render_hop(q, h["hop"]))
    EMB = encode(texts, f"sentence-transformers/{TAG}")
    k = 0
    for q in qs:
        for h in q["hops"]:
            gold = next((d for d in q["docs"] if d["idx"] == h["para_idx"]), None)
            if gold is None:
                k += 1
                continue
            gi = keys.index((gold["title"], gold["text"]))
            bm_scores = np.array(idx.score_docs(texts[k]))
            dn = EMB[k] @ D.T
            items.append({"qid": q["qid"], "hop": h["hop"], "text": texts[k],
                          "rank_dense_bound": rank_of_row(dn, gi),
                          "rank_bm25_bound": rank_of_row(bm_scores, gi)})
            k += 1

    out = {"n_hops": len(items), "hops": items}
    for name in ["rank_dense_bound", "rank_bm25_bound"]:
        by_pos = {}
        for it in items:
            by_pos.setdefault(it["hop"], []).append(it[name])
        out[name] = {p: {"n": len(v), "recall@50": sum(1 for r in v if r <= 50) / len(v),
                         "recall@10": sum(1 for r in v if r <= 10) / len(v)}
                     for p, v in by_pos.items()}
    # de-duplicate BM25 index construction note: bm_scores uses the bound query
    with open(os.path.join(RES, "e2c_hopquery.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nbound requirement query:")
    print("| hop | dense recall@50 | bm25 recall@50 |")
    for p in sorted(out["rank_dense_bound"]):
        print(f"| {p} | {out['rank_dense_bound'][p]['recall@50']:.3f} | "
              f"{out['rank_bm25_bound'][p]['recall@50']:.3f} |")
    print(f"wrote {os.path.join(RES, 'e2c_hopquery.json')}")


if __name__ == "__main__":
    run()
