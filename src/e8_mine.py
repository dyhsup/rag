"""E8: hard-example mining for the distilled gap matcher.

For every hop, rank all chunks with the dense retriever and mine:
  hard_pos  : gold chunk ranked in (3, 50]
  easy_pos  : gold chunk ranked in top-3
  hard_neg  : non-gold chunk ranked ABOVE the gold
  easy_neg  : non-gold chunk just below the gold
Only metadata + text are written; teacher scoring happens in a separate stage.
"""
import argparse
import json
import os
import re

import numpy as np

from .ce_client import CERemote
from .e2b_ce import render_hop
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def chunks_of(text, width=40, stride=20):
    toks = text.split()
    if len(toks) <= width:
        return [" ".join(toks)] if len(toks) >= 8 else []
    out = []
    for i in range(0, len(toks) - width + 1, stride):
        out.append(" ".join(toks[i:i + width]))
    if (len(toks) - width) % stride:
        out.append(" ".join(toks[-width:]))
    return out


def build(questions):
    corpus, meta, gold = [], [], {}
    for q in questions:
        for d in q["docs"]:
            for ci, c in enumerate(chunks_of(d["text"])):
                corpus.append(c)
                meta.append({"qid": q["qid"], "para_idx": d["idx"], "ci": ci})
        for h in q["hops"]:
            ans = norm(h["answer"])
            ids = [i for i, m in enumerate(meta)
                   if m["qid"] == q["qid"] and m["para_idx"] == h["para_idx"]
                   and ans and re.search(r"\b" + re.escape(ans) + r"\b", norm(corpus[i]))]
            gold[(q["qid"], h["hop"])] = ids
    return corpus, meta, gold


def main(n=400, hard_pos=2, hard_neg=3, easy_neg=1, data="musique400.jsonl"):
    qs = load(os.path.join(ROOT, "data", "raw", data))[:n]
    corpus, meta, gold_by_hop = build(qs)
    print(f"[e8] questions={len(qs)} chunks={len(corpus)} hops={len(gold_by_hop)}")

    path = os.path.join(IDX, f"e8_chunks_{len(corpus)}.npy")
    ce = CERemote()
    if os.path.exists(path):
        D = np.load(path)
    else:
        print("[e8] embedding chunks")
        D = ce.embed(corpus)
        np.save(path, D)

    hop_rows = []
    for q in qs:
        for h in q["hops"]:
            hop_rows.append({"qid": q["qid"], "hop": h["hop"], "text": render_hop(q, h["hop"])})
    Q = ce.embed([x["text"] for x in hop_rows])

    out = os.path.join(RES, "e8_mined.jsonl")
    stats = {"hops": 0, "hops_with_gold": 0, "hard_pos": 0, "easy_pos": 0,
             "hard_neg": 0, "easy_neg": 0, "unreachable": 0}
    with open(out, "w", encoding="utf-8") as f:
        for i, hr in enumerate(hop_rows):
            key = (hr["qid"], hr["hop"])
            ids = set(gold_by_hop.get(key, []))
            stats["hops"] += 1
            if not ids:
                continue
            stats["hops_with_gold"] += 1
            s = D @ Q[i]
            order = np.argsort(-s)
            rank_of = {}
            for r, idx in enumerate(order, 1):
                if idx in ids:
                    rank_of[int(idx)] = r
            if not rank_of:
                continue
            gold_rank = min(rank_of.values())
            if gold_rank > 50:
                stats["unreachable"] += 1

            picks = []
            for idx, r in sorted(rank_of.items(), key=lambda kv: kv[1]):
                kind = "hard_pos" if 3 < r <= 50 else ("easy_pos" if r <= 3 else None)
                if kind:
                    picks.append((idx, r, kind, 1))
                    if kind == "hard_pos" and sum(1 for p in picks if p[2] == "hard_pos") >= hard_pos:
                        continue
            # hard negatives: ranked above the gold, not gold
            hn = [int(idx) for idx in order[:gold_rank - 1] if int(idx) not in ids][:hard_neg]
            for idx in hn:
                picks.append((idx, int(np.where(order == idx)[0][0]) + 1, "hard_neg", 0))
            # easy negatives: just below the gold
            below = [int(idx) for idx in order[gold_rank:gold_rank + 50] if int(idx) not in ids][:easy_neg]
            for idx in below:
                picks.append((idx, int(np.where(order == idx)[0][0]) + 1, "easy_neg", 0))

            for idx, r, kind, label in picks:
                rec = {"qid": hr["qid"], "hop": int(hr["hop"]), "idx": int(idx),
                       "q": hr["text"], "p": corpus[idx][:1200], "kind": kind,
                       "label": label, "dense_rank": int(r), "gold_rank": int(gold_rank)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[kind] += 1

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    with open(os.path.join(RES, "e8_mine_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[e8] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--hard-pos", type=int, default=2)
    ap.add_argument("--hard-neg", type=int, default=3)
    ap.add_argument("--easy-neg", type=int, default=1)
    ap.add_argument("--data", default="musique400.jsonl")
    a = ap.parse_args()
    main(a.n, a.hard_pos, a.hard_neg, a.easy_neg, a.data)
