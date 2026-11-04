"""E2b on the remote CE service: does coverage selection beat pointwise rerank?

Candidates = dense top-k. Selectors:
  dense pointwise | CE pointwise (question, doc) | CE coverage (estimated, tau) | oracle coverage
Estimated coverage uses sigmoid(logit) >= tau; evaluation always uses gold labels.
"""
import argparse
import json
import math
import os

import numpy as np

from . import eval as ev
from .ce_client import CERemote
from .e1 import hop_gold, greedy, pointwise
from .e2b_ce import render_hop
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
KS = [1, 2, 3, 4, 6]
TAUS = [0.3, 0.5, 0.7]
TAG = "all-MiniLM-L6-v2"


def run(n=60, cand=50, url="http://10.52.100.210:8833"):
    corpus_qs = load()
    questions = corpus_qs[:n]
    D = np.load(os.path.join(IDX, f"doc_{TAG}.npy"))
    Q = np.load(os.path.join(IDX, f"q_{TAG}.npy"))[:n]
    S = Q @ D.T

    docs, keys = [], []
    for q in corpus_qs:
        for d in q["docs"]:
            docs.append(d["text"])
            keys.append((d["title"], d["text"]))
    assert len(keys) == D.shape[0]

    ce = CERemote(url)
    acc = {"dense_pointwise": [], "ce_pointwise": [], "oracle_coverage": []}
    acc_tau = {t: [] for t in TAUS}
    cand_recall, per_q = [], []

    for qi, q in enumerate(questions):
        order = list(np.argsort(-S[qi])[:cand])
        sub_dense = [float(S[qi][i]) for i in order]
        texts = [docs[i] for i in order]
        gold = hop_gold(q)
        key2hop = {v: h for h, v in gold.items()}
        covers = [({key2hop[keys[i]]} if keys[i] in key2hop else set()) for i in order]
        n_hops = len(gold)
        cand_recall.append(len(set().union(*covers)) / n_hops)

        pw = ce.rerank(q["question"], texts)
        hop = {h: ce.rerank(render_hop(q, h), texts) for h in gold}
        probs = {h: [1 / (1 + math.exp(-x)) for x in hop[h]] for h in hop}
        doc_req = [max(probs[h][i] for h in probs) for i in range(len(order))]

        row = {"qid": q["qid"], "n_hops": n_hops, "cand_recall": cand_recall[-1], "per_k": {}}
        for k in KS:
            s_dense = pointwise(sub_dense, k)
            s_pw = pointwise(pw, k)
            s_or = greedy(sub_dense, covers, k)
            c_dense = len(set().union(*[covers[i] for i in s_dense])) / n_hops
            c_pw = len(set().union(*[covers[i] for i in s_pw])) / n_hops
            c_or = len(set().union(*[covers[i] for i in s_or])) / n_hops
            acc["dense_pointwise"].append((k, c_dense))
            acc["ce_pointwise"].append((k, c_pw))
            acc["oracle_coverage"].append((k, c_or))
            row["per_k"][k] = {"dense_pointwise": c_dense, "ce_pointwise": c_pw,
                               "oracle_coverage": c_or}
            for t in TAUS:
                est = [{h for h in probs if probs[h][i] >= t} for i in range(len(order))]
                sel = greedy(doc_req, est, k)
                c = len(set().union(*[covers[i] for i in sel])) / n_hops
                acc_tau[t].append((k, c))
                row["per_k"][k][f"ce_coverage@{t}"] = c
        per_q.append(row)
        print(f"[{qi+1}/{len(questions)}] {q['qid']} hops={n_hops} "
              f"cand_recall={cand_recall[-1]:.2f}", flush=True)

    def agg(pairs):
        return {k: {"mean": ev.mean([v for kk, v in pairs if kk == k]),
                    "ci": ev.bootstrap_ci([v for kk, v in pairs if kk == k])} for k in KS}

    out = {"n_questions": len(questions), "cand": cand, "url": url,
           "cand_recall": {"mean": ev.mean(cand_recall), "ci": ev.bootstrap_ci(cand_recall)},
           "curve": {m: agg(v) for m, v in acc.items()},
           "ce_coverage_by_tau": {t: agg(v) for t, v in acc_tau.items()},
           "per_question": per_q}
    path = os.path.join(RES, "e2b_remote.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\ncandidate recall@{cand} = {out['cand_recall']['mean']:.3f}")
    hdr = "| k | dense pointwise | CE pointwise | " + " | ".join(f"CE cov@{t}" for t in TAUS) + " | oracle |"
    print(hdr)
    for k in KS:
        cells = [f"{out['curve']['dense_pointwise'][k]['mean']:.3f}",
                 f"{out['curve']['ce_pointwise'][k]['mean']:.3f}"]
        cells += [f"{out['ce_coverage_by_tau'][t][k]['mean']:.3f}" for t in TAUS]
        cells += [f"{out['curve']['oracle_coverage'][k]['mean']:.3f}"]
        print("| " + str(k) + " | " + " | ".join(cells) + " |")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--url", default="http://10.52.100.210:8833")
    a = ap.parse_args()
    run(a.n, a.cand, a.url)
