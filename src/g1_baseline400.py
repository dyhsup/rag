"""IRCoT baseline on the same 400-question pool as e13 ctxlast."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .e4_pipeline import run_ircot
from .musique import load
from .retriever import QuestionIndex
from .llm import LLM
from .ce_client import CERemote

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def main(n=400, workers=10, data="musique400.jsonl"):
    qs = load(os.path.join(ROOT, "data", "raw", data))[:n]
    docs = []
    for q in qs:
        for d in q["docs"]:
            docs.append({"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
                         "sentences": d["sentences"]})
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()
    path = os.path.join(RES, f"e4_{data.replace(chr(46)+'jsonl','')}_ircot.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[g1base] {len(todo)} to run")

    def work(q):
        pred, info = run_ircot(q, index, llm, ce)
        from .e3_binding import em, f1
        rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
               "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"]), "info": info}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 40 == 0 or i == len(todo):
                print(f"[g1base] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--data", default="musique400.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
