"""Counterfactual Marginal Utility (CMU) evidence selection.

Delta(d|S) = log p(a | q, S∪{d}) - log p(a | q, S), measured with the generator
via /v1/completions echo+logprobs. Greedy selection maximizes Delta.
Compared against CE pointwise selection on the same pool.
"""
import argparse, json, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
EPS = ["http://10.94.199.216:8000/v1", "http://10.94.199.217:8000/v1"]
MODEL = "qwen3.6-27b"
_lock = threading.Lock()
_rr = [0]


def logp_answer(q, passages, answer):
    body = "\n".join(f"[{i+1}] {p[:400]}" for i, p in enumerate(passages)) or "(none)"
    prompt = f"Question: {q}\nPassages:\n{body}\nAnswer: {answer}"
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 0,
               "echo": True, "logprobs": 0, "temperature": 0}
    for attempt in range(3):
        with _lock:
            ep = EPS[_rr[0] % len(EPS)]; _rr[0] += 1
        try:
            r = requests.post(ep + "/completions", json=payload, proxies={}, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(r.text[:120])
            lp = r.json()["choices"][0]["logprobs"]
            cut = prompt.rfind("Answer: ") + len("Answer: ")
            tot = sum(t for o, t in zip(lp["text_offset"], lp["token_logprobs"])
                      if o >= cut and t is not None)
            return tot
        except Exception:
            continue
    return float("-inf")


def main(n=100, workers=6, k=2):
    qs = load()[:n]
    ce = CERemote(); llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    path = os.path.join(RES, "e21_cmu.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[e21] {len(todo)} questions, k={k}")

    def work(q):
        docs = [d["text"] for d in q["docs"]][:20]
        gold = [d["text"] for d in q["docs"] if d["is_gold"]]
        base = logp_answer(q["question"], [], q["answer"])
        # CE pointwise on the same pool
        try:
            csc = ce.rerank(q["question"], docs)
            ce_idx = sorted(range(len(docs)), key=lambda i: -csc[i])[:k]
        except Exception:
            ce_idx = list(range(k))
        # CMU greedy
        sel, cur = [], base
        for _ in range(k):
            best, best_d, best_lp = -1e9, None, cur
            for i, d in enumerate(docs):
                if i in sel:
                    continue
                lp = logp_answer(q["question"], [docs[j] for j in sel] + [d], q["answer"])
                if lp - cur > best:
                    best, best_d, best_lp = lp - cur, i, lp
            if best_d is None:
                break
            sel.append(best_d); cur = best_lp
        def read_em(idx):
            ps = [docs[i] for i in idx]
            try:
                pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(ps))}],
                                max_tokens=32).strip().split("\n")[0]
            except Exception:
                pred = ""
            return pred
        p_ce, p_cmu = read_em(ce_idx), read_em(sel)
        rec = {"qid": q["qid"], "gold": q["answer"],
               "ce_idx": ce_idx, "cmu_idx": sel,
               "ce_gold": any(docs[i] in gold for i in ce_idx),
               "cmu_gold": any(docs[i] in gold for i in sel),
               "ce_pred": p_ce, "cmu_pred": p_cmu,
               "ce_em": em(p_ce, q["answer"]), "cmu_em": em(p_cmu, q["answer"]),
               "ce_f1": f1(p_ce, q["answer"]), "cmu_f1": f1(p_cmu, q["answer"])}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 10 == 0 or i == len(todo):
                print(f"[e21] {i}/{len(todo)}", flush=True)

    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(json.dumps({
        "n": len(recs),
        "gold_in_set": {"CE": sum(r["ce_gold"] for r in recs) / len(recs),
                        "CMU": sum(r["cmu_gold"] for r in recs) / len(recs)},
        "EM": {"CE": sum(r["ce_em"] for r in recs) / len(recs),
               "CMU": sum(r["cmu_em"] for r in recs) / len(recs)},
        "F1": {"CE": sum(r["ce_f1"] for r in recs) / len(recs),
               "CMU": sum(r["cmu_f1"] for r in recs) / len(recs)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    main(a.n, a.workers, a.k)
