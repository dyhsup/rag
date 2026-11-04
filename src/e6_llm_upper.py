"""E6: upper bound on semantic-gap closure with an LLM pairwise judge.

For every gap case (gold in dense top-50 but outside top-3), take the dense
top-20 and rerank with (a) the cross-encoder and (b) an LLM sufficiency judge.
Both are evaluated on the same candidate set, so the comparison is fair.
"""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .ce_client import CERemote
from .e5_gap import build, measure
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_lock = threading.Lock()
_LLM = None

JUDGE = """Does the passage state the fact needed to answer the question? Use only the passage.
Reply with a single integer from 0 (no) to 10 (fully states it). No explanation.

Question: {q}
Passage: {p}
Score:"""


def parse_int(text):
    m = re.search(r"\b(10|[0-9])\b", text or "")
    return int(m.group(1)) if m else 0


def run(n=200, topn=20, workers=8):
    qs = load()[:n]
    ce = CERemote()
    rows, corpus, meta, emb = measure(qs, ce)
    import src.e5_gap as M
    M._GOLD = build(qs)[2]
    gold = M._GOLD

    gaps = [r for r in rows if r["rank_dense"] and 3 < r["rank_dense"] <= 50]
    in_top20 = sum(1 for r in gaps if r["rank_dense"] <= topn)
    print(f"[e6] gap cases={len(gaps)}  gold in dense top-{topn}: {in_top20}")

    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    jobs = []
    for gi, r in enumerate(gaps):
        top = list(np.argsort(-r["dense_scores"])[:topn])
        for j, idx in enumerate(top):
            jobs.append((gi, j, idx, r["text"], corpus[idx]))
    print(f"[e6] LLM pairs={len(jobs)}")

    scores = [[None] * topn for _ in gaps]
    def work(job):
        gi, j, idx, q, p = job
        try:
            return gi, j, parse_int(llm.chat(
                [{"role": "user", "content": JUDGE.format(q=q, p=p[:1200])}], max_tokens=8))
        except Exception:
            return gi, j, 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            gi, j, s = fut.result()
            scores[gi][j] = s
            if i % 200 == 0 or i == len(jobs):
                print(f"[e6] {i}/{len(jobs)} calls={llm.calls}", flush=True)

    closed_llm = closed_ce = 0
    details = []
    for gi, r in enumerate(gaps):
        ids = gold[(r["qid"], r["hop"])]
        top = list(np.argsort(-r["dense_scores"])[:topn])
        texts = [corpus[i] for i in top]

        ce_scores = ce.rerank(r["text"], texts)
        order_ce = sorted(range(topn), key=lambda j: -float(ce_scores[j]))
        rk_ce = next((p + 1 for p, j in enumerate(order_ce) if top[j] in ids), None)

        order_llm = sorted(range(topn), key=lambda j: (-(scores[gi][j] or 0), -float(ce_scores[j])))
        rk_llm = next((p + 1 for p, j in enumerate(order_llm) if top[j] in ids), None)

        closed_ce += int(rk_ce is not None and rk_ce <= 3)
        closed_llm += int(rk_llm is not None and rk_llm <= 3)
        details.append({"qid": r["qid"], "hop": r["hop"], "before": r["rank_dense"],
                        "ce_rank": rk_ce, "llm_rank": rk_llm,
                        "llm_scores": scores[gi]})

    out = {"n_questions": n, "topn": topn, "gap_cases": len(gaps),
           "gold_in_topn": in_top20,
           "closure_ce": closed_ce, "closure_rate_ce": closed_ce / len(gaps),
           "closure_llm": closed_llm, "closure_rate_llm": closed_llm / len(gaps),
           "details": details}
    with open(os.path.join(RES, "e6_llm_upper.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run(a.n, a.topn, a.workers)
