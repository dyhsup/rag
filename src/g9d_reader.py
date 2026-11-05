"""Single-hop: reader protocol variants (evidence count + locate-then-extract)."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt, topk
from .llm import LLM
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

COT = """Answer the question from the passages.
Step 1: find the sentence that states the answer.
Step 2: extract the shortest answer span.
Output only the answer span, nothing else.

Question: {q}
Passages:
{ps}
Answer:"""


def main(n=200, workers=10, data="squad_200.jsonl", modes=("top1", "top3", "cot5")):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", data))][:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()
    tag = data.replace(".jsonl", "")

    for mode in modes:
        k = {"top1": 1, "top3": 3, "cot5": 5}[mode]
        prompt = COT if mode == "cot5" else READ
        path = os.path.join(RES, f"g9d_{tag}_{mode}.jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                done.add(json.loads(line)["qid"])
        todo = [q for q in qs if q["qid"] not in done]
        print(f"[g9d] {tag} {mode}: {len(todo)}")

        def work(q, k=k, prompt=prompt):
            cand, _ = topk(index, q["question"], 50)
            try:
                sc = ce.rerank(q["question"], cand)
                order = sorted(range(len(cand)), key=lambda i: -sc[i])[:k]
                ps = [cand[i] for i in order]
            except Exception:
                ps = cand[:k]
            try:
                pred = llm.chat([{"role": "user", "content": prompt.format(q=q["question"], ps=fmt(ps))}],
                                max_tokens=48).strip().split("\n")[0]
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
                if i % 50 == 0 or i == len(todo):
                    print(f"[g9d] {mode} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="squad_200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
