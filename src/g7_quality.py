"""QuALITY long-document MCQ baseline."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .e3_binding import em, f1
from .e4_pipeline import fmt, topk
from .llm import LLM
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

PROMPT = """Answer the multiple-choice question using only the passages.
Output exactly the text of one option, nothing else.

Question: {q}
Options:
{opts}
Passages:
{ps}
Answer:"""


def main(n=200, workers=10):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", "quality_200.jsonl"))][:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    path = os.path.join(RES, "g7_quality.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]

    def work(q):
        ps, _ = topk(index, q["question"], 5)
        opts = "\n".join(f"- {o}" for o in q["options"])
        try:
            pred = llm.chat([{"role": "user", "content": PROMPT.format(q=q["question"], opts=opts, ps=fmt(ps))}],
                            max_tokens=32).strip().split("\n")[0]
        except Exception:
            pred = ""
        # map to the closest option
        best, score = pred, -1
        for o in q["options"]:
            s = len(set(pred.lower().split()) & set(o.lower().split()))
            if s > score:
                best, score = o, s
        rec = {"qid": q["qid"], "pred": best, "raw": pred, "gold": q["answer"],
               "em": em(best, q["answer"]), "f1": f1(best, q["answer"]),
               "retrieved_gold": any(d["is_gold"] for d in q["docs"] if d["text"] in ps)}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 50 == 0 or i == len(todo):
                print(f"[g7] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
