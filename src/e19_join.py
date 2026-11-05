"""Binding-aware join: structured facts + evidence -> final aggregation."""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import fmt, topk as bm25_topk
from .llm import LLM
from .retriever import QuestionIndex
from .musique import load
from .data import load_questions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

READ_COT = """Answer the sub-question using only the passages.
Step 1: find the sentence that states the answer.
Step 2: extract the shortest answer span.
Output only the answer span.

Sub-question: {q}
Passages:
{ps}
Answer:"""

JOIN = """Answer the question using the grounded facts below. Each fact was extracted from the cited evidence.

Rules:
- If the question compares two entities, compare their grounded values and output the requested entity, or yes/no.
- If the question asks for a judgement, decide from the facts.
- Use only the facts; if a fact is missing, say "insufficient".
- Output only the final answer.

Question: {q}
Facts:
{facts}
Answer:"""

_cache = {"emb": {}}


def retrieve(ce, index, text, k=5):
    key = (id(index), len(index.docs))
    if key not in _cache["emb"]:
        _cache["emb"][key] = ce.embed([d["text"] for d in index.docs])
    de = _cache["emb"][key]
    cbm, _ = bm25_topk(index, text, 20)
    try:
        qe = ce.embed([text])[0]
        order = list(np.argsort(-(de @ qe)))[:20]
        cdn = [index.docs[i]["text"] for i in order]
    except Exception:
        cdn = []
    cand = list(dict.fromkeys(cbm + cdn))
    try:
        sc = ce.rerank(text, cand)
        return [cand[i] for i in sorted(range(len(cand)), key=lambda i: -sc[i])[:k]]
    except Exception:
        return cand[:k]


def render(text, binds):
    for i, a in enumerate(binds, 1):
        if a:
            text = text.replace(f"#{i}", a)
    return re.sub(r"#\d+", " ", text).replace(">>", " ").strip()


def load_data(path):
    p = os.path.join(ROOT, "data", "raw", path)
    return load_questions(p) if path.startswith("hotpot") else load(p)


def main(n=200, workers=10, data="wiki2_200.jsonl"):
    qs = load_data(data)[:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    ce = CERemote(); llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    tag = data.replace(".jsonl", "")

    plans = {}
    pp = os.path.join(RES, f"e14_plans_{tag}.jsonl")
    if os.path.exists(pp):
        for line in open(pp, encoding="utf-8"):
            r = json.loads(line); plans[r["qid"]] = r["hops"]

    path = os.path.join(RES, f"e19_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[e19] {tag}: {len(todo)}")

    def work(q):
        hops = plans.get(q["qid"]) or [{"query": q["question"], "type": "other"}]
        binds, facts = [], []
        for h, hop in enumerate(hops[:4]):
            text = render(hop["query"], binds)
            ps = retrieve(ce, index, text)
            try:
                a = llm.chat([{"role": "user", "content": READ_COT.format(q=text, ps=fmt(ps))}],
                             max_tokens=32).strip().split("\n")[0]
            except Exception:
                a = ""
            facts.append(f"F{h+1}. sub-question: {text} | answer: {a} | evidence: {ps[0][:180] if ps else ''}")
            binds.append(a)
        try:
            pred = llm.chat([{"role": "user", "content": JOIN.format(q=q["question"], facts="\n".join(facts))}],
                            max_tokens=32).strip().split("\n")[0]
        except Exception:
            pred = ""
        rec = {"qid": q["qid"], "type": q.get("type", ""), "pred": pred, "gold": q["answer"],
               "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"])}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 25 == 0 or i == len(todo):
                print(f"[e19] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="wiki2_200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
