"""E2b: off-the-shelf cross-encoder as the grounding matcher.

Candidates = dense top-100. Three selectors on the same candidate set:
  - dense pointwise (baseline)
  - CE pointwise (question, passage) rerank
  - CE coverage (per-requirement scores + greedy coverage)
Plus oracle coverage as the ceiling for this candidate set.
"""
import argparse
import json
import os

import math

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import eval as ev
from .e1 import hop_gold, greedy, pointwise
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
KS = [1, 2, 3, 4, 6]
TAUS = [0.3, 0.5, 0.7]
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def render_hop(q, h):
    """MuSiQue hop text with previous-hop answers substituted."""
    step = q["hops"][h]
    text = step["question"]
    for j, prev in enumerate(q["hops"][:h], 1):
        text = text.replace(f"#{j}", prev["answer"])
    return text.replace(">>", " ").strip()


def load_ce(name=CE_MODEL):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
    torch.set_num_threads(8)
    return tok, mdl


def score_pairs(pairs, tok, mdl, batch=64, tag=""):
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            b = tok([p[0] for p in pairs[i:i + batch]],
                    [p[1] for p in pairs[i:i + batch]],
                    padding=True, truncation=True, max_length=192, return_tensors="pt")
            logits = mdl(**b).logits.view(-1)
            out.extend(logits.tolist())
            if (i // batch) % 100 == 0:
                print(f"  [{tag}] {i + len(b['input_ids'])}/{len(pairs)}", flush=True)
    return out


def run(n=200, cand=100, model="sentence-transformers/all-MiniLM-L6-v2", limit_q=None):
    corpus_qs = load()  # the pooled corpus must cover every row of D
    questions = corpus_qs[:n]
    tag = model.split("/")[-1]
    D = np.load(os.path.join(IDX, f"doc_{tag}.npy"))
    Q = np.load(os.path.join(IDX, f"q_{tag}.npy"))[:n]
    S = Q @ D.T

    docs, keys = [], []
    for q in corpus_qs:
        for d in q["docs"]:
            docs.append(d["text"])
            keys.append((d["title"], d["text"]))
    assert len(keys) == D.shape[0], (len(keys), D.shape)

    tok, mdl = load_ce()

    ce_pw, ce_hop, dense_pw, oracle_cov, cand_recall = [], [], [], [], []
    per_q = []
    for qi, q in enumerate(questions):
        order = list(np.argsort(-S[qi])[:cand])
        sub_dense = [float(S[qi][i]) for i in order]
        gold = hop_gold(q)
        key2hop = {v: h for h, v in gold.items()}
        covers = [({key2hop[keys[i]]} if keys[i] in key2hop else set()) for i in order]
        n_hops = len(gold)
        cand_recall.append(len(set().union(*covers)) / n_hops)

        # CE pointwise: question vs passage
        pw_scores = score_pairs([(q["question"], docs[i]) for i in order], tok, mdl,
                                tag=f"pw {qi}")
        # CE per requirement: hop text vs passage
        hop_scores = {}
        for h in gold:
            hop_scores[h] = score_pairs([(render_hop(q, h), docs[i]) for i in order], tok, mdl,
                                        tag=f"hop {qi}.{h}")

        row = {"qid": q["qid"], "n_hops": n_hops, "per_k": {}}
        probs = {h: [1.0 / (1.0 + math.exp(-v)) for v in hop_scores[h]] for h in hop_scores}
        doc_req = [max(probs[h][i] for h in probs) for i in range(len(order))]
        for k in KS:
            sel_pw = pointwise(pw_scores, k)
            sel_dense = pointwise(sub_dense, k)
            sel_oracle = greedy(sub_dense, covers, k)
            cov_ce_pw = len(set().union(*[covers[i] for i in sel_pw])) / n_hops
            cov_dense = len(set().union(*[covers[i] for i in sel_dense])) / n_hops
            cov_oracle = len(set().union(*[covers[i] for i in sel_oracle])) / n_hops
            ce_pw.append((k, cov_ce_pw))
            dense_pw.append((k, cov_dense))
            oracle_cov.append((k, cov_oracle))
            row["per_k"][k] = {"dense_pointwise": cov_dense, "ce_pointwise": cov_ce_pw,
                               "oracle_coverage": cov_oracle}
            # coverage estimated from CE scores, evaluated with gold labels
            for tau in TAUS:
                covers_est = [{h for h in probs if probs[h][i] >= tau} for i in range(len(order))]
                sel = greedy(doc_req, covers_est, k)
                cov = len(set().union(*[covers[i] for i in sel])) / n_hops
                ce_hop.append((tau, k, cov))
                row["per_k"][k][f"ce_coverage@{tau}"] = cov
        per_q.append(row)
        if qi % 10 == 0:
            print(f"question {qi}/{len(questions)}", flush=True)

    def agg(pairs):
        return {k: {"mean": ev.mean([v for kk, v in pairs if kk == k]),
                    "ci": ev.bootstrap_ci([v for kk, v in pairs if kk == k])} for k in KS}

    def agg_tau(pairs):
        return {tau: agg([(k, v) for tt, k, v in pairs if tt == tau]) for tau in TAUS}

    out = {
        "n_questions": len(questions), "cand": cand, "ce_model": CE_MODEL,
        "cand_recall": {"mean": ev.mean(cand_recall), "ci": ev.bootstrap_ci(cand_recall)},
        "curve": {"dense_pointwise": agg(dense_pw), "ce_pointwise": agg(ce_pw),
                  "oracle_coverage": agg(oracle_cov)},
        "ce_coverage_by_tau": agg_tau(ce_hop),
        "per_question": per_q,
    }
    path = os.path.join(RES, "e2b_ce.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\ncandidate recall@{cand} = {out['cand_recall']['mean']:.3f}")
    print("| k | dense pointwise | CE pointwise | CE coverage@0.5 | oracle coverage |")
    for k in KS:
        c = out["curve"]
        print(f"| {k} | {c['dense_pointwise'][k]['mean']:.3f} | {c['ce_pointwise'][k]['mean']:.3f} | "
              f"{out['ce_coverage_by_tau'][0.5][k]['mean']:.3f} | {c['oracle_coverage'][k]['mean']:.3f} |")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cand", type=int, default=100)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    run(a.n, a.cand, a.model)
