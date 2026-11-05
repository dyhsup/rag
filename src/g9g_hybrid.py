"""Single-hop: hybrid retrieval (BM25 union dense) + CE + locate-then-extract."""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import fmt, topk
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


def main(n=200, workers=10, data="squad_200.jsonl", k=5):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", data))][:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    ce = CERemote(); llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    tag = data.replace(".jsonl", "")
    path = os.path.join(RES, f"g9g_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[g9g] {tag}: {len(todo)}")

    def work(q):
        texts = [d["text"] for d in q["docs"]]
        gold = [d["text"] for d in q["docs"] if d["is_gold"]]
        bm, _ = topk(index, q["question"], 20)
        try:
            de = ce.embed(texts); qe = ce.embed([q["question"]])[0]
            order = list(np.argsort(-(de @ qe)))[:20]
            dn = [texts[i] for i in order]
        except Exception:
            dn = []
        cand = list(dict.fromkeys(bm + dn))[:60]
        try:
            cs = ce.rerank(q["question"], cand)
            o = sorted(range(len(cand)), key=lambda i: -cs[i])[:k]
            ev = [cand[i] for i in o]
        except Exception:
            ev = cand[:k]
        try:
            pred = llm.chat([{"role": "user", "content": COT.format(q=q["question"], ps=fmt(ev))}],
                            max_tokens=48).strip().split("\n")[0]
        except Exception:
            pred = ""
        rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
               "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"]),
               "gold_in_topk": any(g in ev for g in gold)}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 50 == 0 or i == len(todo):
                print(f"[g9g] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="squad_200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
