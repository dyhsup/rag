"""HyDE end-to-end baseline on the same MuSiQue pool."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .e3_binding import em, f1
from .e4_pipeline import READ, fmt, topk
from .llm import LLM
from .musique import load
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def main(n=200, workers=10):
    qs = load()[:n]
    docs = []
    for q in qs:
        for d in q["docs"]:
            docs.append({"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
                         "sentences": d["sentences"]})
    index = QuestionIndex({"qid": "pool", "docs": docs})
    pseudo = {}
    for line in open(os.path.join(RES, "hyde_pseudo.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        pseudo[r["qid"]] = r["text"]
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    path = os.path.join(RES, "g4_hyde.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[hyde] {len(todo)} to run")

    def work(q):
        text = pseudo.get(q["qid"]) or q["question"]
        ps, _ = topk(index, text, 5)
        try:
            pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(ps))}],
                            max_tokens=32).strip().split("\n")[0]
        except Exception:
            pred = ""
        rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
               "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"])}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 40 == 0 or i == len(todo):
                print(f"[hyde] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
