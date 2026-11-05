"""Single-hop: multi-passage reading with agreement-based answer selection.

For one requirement, the program still grounds over several evidence items:
read each top-5 passage separately, aggregate by normalized agreement, and break
ties with the CE score of the supporting passage.
"""
import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt, topk
from .llm import LLM
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def retrieve(index, ce, text, k=5, pool=50):
    cand, _ = topk(index, text, pool)
    try:
        sc = ce.rerank(text, cand)
    except Exception:
        return cand[:k], [0.0] * len(cand[:k])
    order = sorted(range(len(cand)), key=lambda i: -sc[i])[:k]
    return [cand[i] for i in order], [sc[i] for i in order]


def select(q, index, ce, llm, mode):
    ps, sc = retrieve(index, ce, q["question"])
    if mode == "concat":
        try:
            a = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(ps))}],
                         max_tokens=32).strip().split("\n")[0]
        except Exception:
            a = ""
        return a
    answers = []
    for p in ps:
        try:
            a = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt([p]))}],
                         max_tokens=32).strip().split("\n")[0]
        except Exception:
            a = ""
        answers.append((a, p))
    cnt = Counter(norm(a) for a, _ in answers if norm(a))
    if cnt:
        best_key, best_n = cnt.most_common(1)[0]
        cands = [(a, p) for a, p in answers if norm(a) == best_key]
        if mode == "vote":
            return cands[0][0]
        # vote + CE tie-break: pick the passage with the highest CE score
        best = max(cands, key=lambda x: sc[ps.index(x[1])] if x[1] in ps else -1e9)
        return best[0]
    return answers[0][0] if answers else ""


def main(n=200, workers=10, data="squad_200.jsonl", modes=("concat", "vote", "vote_ce")):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", data))][:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()
    tag = data.replace(".jsonl", "")
    for mode in modes:
        path = os.path.join(RES, f"g9_{tag}_{mode}.jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                done.add(json.loads(line)["qid"])
        todo = [q for q in qs if q["qid"] not in done]
        print(f"[g9] {tag} {mode}: {len(todo)}")

        def work(q, mode=mode):
            pred = select(q, index, ce, llm, mode)
            rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
                   "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"])}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
                fut.result()
                if i % 50 == 0 or i == len(todo):
                    print(f"[g9] {mode} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="squad_200.jsonl")
    ap.add_argument("--modes", default="concat,vote,vote_ce")
    a = ap.parse_args()
    main(a.n, 10, a.data, tuple(a.modes.split(",")))
