"""Single-hop baselines on the SQuAD pool: vanilla and IRCoT."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt, run_ircot, topk
from .llm import LLM
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def main(n=200, workers=10, data="squad_200.jsonl"):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", data))][:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()

    for method in ["vanilla", "ircot"]:
        tag = data.replace(".jsonl", "")
        path = os.path.join(RES, f"g5_{tag}_{method}.jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                done.add(json.loads(line)["qid"])
        todo = [q for q in qs if q["qid"] not in done]
        print(f"[g5] {method}: {len(todo)}")

        def work(q, method=method):
            if method == "vanilla":
                ps, _ = topk(index, q["question"], 5)
                try:
                    pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(ps))}],
                                    max_tokens=32).strip().split("\n")[0]
                except Exception:
                    pred = ""
            else:
                pred, _ = run_ircot(q, index, llm, ce)
            rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
                   "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"])}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
                fut.result()
                if i % 50 == 0 or i == len(todo):
                    print(f"[g5] {method} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="squad_200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
