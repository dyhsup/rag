"""E2a baseline: does HyDE raise the first-stage ceiling?

Generate a hypothetical answer passage for each question, encode it instead of
the question, and re-measure hop-level candidate recall on the same MuSiQue
pooled corpus.
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import eval as ev
from .e1 import hop_gold, greedy, pointwise
from .e1_dense import encode
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
KS = [1, 2, 3, 4, 6]
_lock = threading.Lock()

PROMPT = """Write a short Wikipedia-style passage (2-3 sentences) that would answer the question.
Do not say "I don't know". Do not add commentary. Output only the passage.

Question: {question}
Passage:"""


def gen(questions, llm, workers=8, path=None):
    path = path or os.path.join(RES, "hyde_pseudo.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["qid"]] = r
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[hyde] {len(todo)} to generate ({len(done)} cached)")

    def work(q):
        try:
            text = llm.chat([{"role": "user", "content": PROMPT.format(question=q["question"])}],
                            max_tokens=200).strip()
        except Exception as e:
            text = q["question"]
        return {"qid": q["qid"], "text": text}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            rec = fut.result()
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[hyde] {i}/{len(todo)} calls={llm.calls} hits={llm.cache_hits}", flush=True)


def score(questions, model_name, cand=50):
    docs, keys = [], []
    for q in questions:
        for d in q["docs"]:
            docs.append(d["text"])
            keys.append((d["title"], d["text"]))
    tag = model_name.split("/")[-1]
    D = np.load(os.path.join(IDX, f"doc_{tag}.npy"))
    pseudo = {}
    with open(os.path.join(RES, "hyde_pseudo.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pseudo[r["qid"]] = r["text"]
    texts = [pseudo.get(q["qid"], q["question"]) for q in questions]
    Q = encode(texts, model_name)  # not cached: pseudo-docs are new
    S = Q @ D.T

    curve = {k: {"pointwise": [], "coverage": []} for k in KS}
    cand_recall = []
    for qi, q in enumerate(questions):
        cand_idx = list(np.argsort(-S[qi])[:cand])
        sub = [float(S[qi][i]) for i in cand_idx]
        gold = hop_gold(q)
        key2hop = {v: h for h, v in gold.items()}
        covers = [({key2hop[keys[i]]} if keys[i] in key2hop else set()) for i in cand_idx]
        n_hops = len(gold)
        cand_recall.append(len(set().union(*covers)) / n_hops)
        for k in KS:
            pw = pointwise(sub, k)
            cv = greedy(sub, covers, k)
            curve[k]["pointwise"].append(len(set().union(*[covers[i] for i in pw])) / n_hops)
            curve[k]["coverage"].append(len(set().union(*[covers[i] for i in cv])) / n_hops)

    out = {"model": model_name, "method": "hyde", "n_questions": len(questions),
           "cand_recall": {"mean": ev.mean(cand_recall), "ci": ev.bootstrap_ci(cand_recall)},
           "curve": {k: {m: {"mean": ev.mean(v), "ci": ev.bootstrap_ci(v)} for m, v in d.items()}
                     for k, d in curve.items()}}
    path = os.path.join(RES, f"e2a_hyde_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nHyDE + {model_name}: candidate recall@{cand} = {out['cand_recall']['mean']:.3f}")
    for k in KS:
        print(f"| {k} | {out['curve'][k]['pointwise']['mean']:.3f} | "
              f"{out['curve'][k]['coverage']['mean']:.3f} |")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["gen", "score", "all"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    qs = load()[: a.n]
    if a.stage in ("gen", "all"):
        gen(qs, LLM(os.path.join(ROOT, "cache", "llm.sqlite")))
    if a.stage in ("score", "all"):
        score(qs, a.model)
