"""E2a: is the first-stage ceiling BM25-specific?

Same MuSiQue pooled corpus and oracle hop labels as E1, but candidates come from
a dense retriever. If dense also stalls well below 1.0, the first-stage gap is
not a BM25 artifact.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import eval as ev
from .e1 import hop_gold, greedy, pointwise
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
KS = [1, 2, 3, 4, 6]


def encode(texts, model_name, batch=32):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    torch.set_num_threads(8)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
            h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            e = torch.nn.functional.normalize(e, dim=-1)
            out.append(e.numpy())
            if (i // batch) % 20 == 0:
                print(f"  encoded {i + len(b['input_ids'])}/{len(texts)}", flush=True)
    return np.vstack(out).astype(np.float32)


def run(n=200, model_name="sentence-transformers/all-MiniLM-L6-v2", cand=50):
    questions = load()[:n]
    docs, keys = [], []
    for q in questions:
        for d in q["docs"]:
            docs.append(d["text"])
            keys.append((d["title"], d["text"]))
    queries = [q["question"] for q in questions]

    os.makedirs(IDX, exist_ok=True)
    tag = model_name.split("/")[-1]
    dpath, qpath = os.path.join(IDX, f"doc_{tag}.npy"), os.path.join(IDX, f"q_{tag}.npy")
    if os.path.exists(dpath):
        D = np.load(dpath)
    else:
        print(f"encoding {len(docs)} docs with {model_name}")
        D = encode(docs, model_name)
        np.save(dpath, D)
    if os.path.exists(qpath):
        Q = np.load(qpath)
    else:
        print(f"encoding {len(queries)} queries")
        Q = encode(queries, model_name)
        np.save(qpath, Q)

    S = Q @ D.T
    curve = {k: {"pointwise": [], "coverage": []} for k in KS}
    cand_recall = []
    for qi, q in enumerate(questions):
        order = np.argsort(-S[qi])[: len(docs)]
        cand_idx = list(order[:cand])
        sub = [float(S[qi][i]) for i in cand_idx]
        gold = hop_gold(q)
        key2hop = {v: h for h, v in gold.items()}
        covers = []
        for i in cand_idx:
            h = key2hop.get(keys[i])
            covers.append({h} if h is not None else set())
        n_hops = len(gold)
        cand_recall.append(len(set().union(*covers)) / n_hops)
        for k in KS:
            pw = pointwise(sub, k)
            cv = greedy(sub, covers, k)
            curve[k]["pointwise"].append(len(set().union(*[covers[i] for i in pw])) / n_hops)
            curve[k]["coverage"].append(len(set().union(*[covers[i] for i in cv])) / n_hops)

    summary = {k: {m: {"mean": ev.mean(v), "ci": ev.bootstrap_ci(v)} for m, v in d.items()}
               for k, d in curve.items()}
    out = {"model": model_name, "n_questions": len(questions), "n_docs": len(docs),
           "cand_recall": {"mean": ev.mean(cand_recall), "ci": ev.bootstrap_ci(cand_recall)},
           "curve": summary}
    path = os.path.join(RES, f"e2a_dense_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{model_name}: candidate recall@{cand} = {out['cand_recall']['mean']:.3f}")
    print("| k | pointwise | greedy |")
    for k in KS:
        print(f"| {k} | {summary[k]['pointwise']['mean']:.3f} | {summary[k]['coverage']['mean']:.3f} |")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    run(args.n, args.model)
