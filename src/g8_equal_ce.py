"""Equalize the reranker: every method gets BM25 top-50 + the same CE top-5."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import IRCOT_NEXT, READ, fmt, topk
from .llm import LLM
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def retrieve_ce(index, ce, text, k=5, pool=50):
    cand, _ = topk(index, text, pool)
    try:
        scores = ce.rerank(text, cand)
    except Exception:
        return cand[:k]
    order = sorted(range(len(cand)), key=lambda i: -scores[i])[:k]
    return [cand[i] for i in order]


def solve(q, index, ce, llm, method):
    if method == "vanilla_ce":
        ps = retrieve_ce(index, ce, q["question"])
        try:
            pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(ps))}],
                            max_tokens=32).strip().split("\n")[0]
        except Exception:
            pred = ""
        return pred
    # ircot_ce
    ps = retrieve_ce(index, ce, q["question"])
    acc = list(ps)
    for _ in range(2):
        try:
            sent = llm.chat([{"role": "user", "content": IRCOT_NEXT.format(q=q["question"], ps=fmt(acc[-5:]))}],
                            max_tokens=48).strip().split("\n")[0]
        except Exception:
            break
        if not sent:
            break
        ps = retrieve_ce(index, ce, sent)
        acc.extend(ps)
    seen, uniq = set(), []
    for p in acc:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    try:
        return llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(uniq[:8]))}],
                        max_tokens=32).strip().split("\n")[0]
    except Exception:
        return ""


def main(n=200, workers=10, data="musique200.jsonl"):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", data))][:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()
    tag = data.replace(".jsonl", "")
    for method in ["vanilla_ce", "ircot_ce"]:
        path = os.path.join(RES, f"g8_{tag}_{method}.jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                done.add(json.loads(line)["qid"])
        todo = [q for q in qs if q["qid"] not in done]
        print(f"[g8] {tag} {method}: {len(todo)}")

        def work(q, method=method):
            pred = solve(q, index, ce, llm, method)
            rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
                   "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"])}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
                fut.result()
                if i % 50 == 0 or i == len(todo):
                    print(f"[g8] {method} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="musique200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
