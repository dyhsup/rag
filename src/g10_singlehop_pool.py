"""Single-hop, per-question 10-passage pool: fair comparison of reader protocols."""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, IRCOT_NEXT, fmt
from .llm import LLM

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


def pick(ce, q, docs, k=5):
    try:
        sc = ce.rerank(q, docs)
        return [docs[i] for i in sorted(range(len(docs)), key=lambda i: -sc[i])[:k]]
    except Exception:
        return docs[:k]


def run_one(q, ce, llm, mode):
    docs = [d["text"] for d in q["docs"]]
    ev = pick(ce, q["question"], docs, 5)
    if mode == "vanilla":
        prompt = READ
    elif mode == "cot":
        prompt = COT
    else:  # ircot on the same pool
        acc = list(ev)
        for _ in range(2):
            try:
                sent = llm.chat([{"role": "user", "content": IRCOT_NEXT.format(q=q["question"], ps=fmt(acc[-5:]))}],
                                max_tokens=48).strip().split("\n")[0]
            except Exception:
                break
            if not sent:
                break
            acc.extend(pick(ce, sent, docs, 3))
        seen, uniq = set(), []
        for p in acc:
            if p not in seen:
                seen.add(p); uniq.append(p)
        prompt = READ
        ev = uniq[:8]
    try:
        pred = llm.chat([{"role": "user", "content": prompt.format(q=q["question"], ps=fmt(ev))}],
                        max_tokens=48).strip().split("\n")[0]
    except Exception:
        pred = ""
    return pred


def main(n=200, workers=10, data="squad_200.jsonl", modes=("vanilla", "cot", "ircot")):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", data))][:n]
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite")); ce = CERemote()
    tag = data.replace(".jsonl", "")
    for mode in modes:
        path = os.path.join(RES, f"g10_{tag}_{mode}.jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                done.add(json.loads(line)["qid"])
        todo = [q for q in qs if q["qid"] not in done]
        print(f"[g10] {tag} {mode}: {len(todo)}")

        def work(q, mode=mode):
            pred = run_one(q, ce, llm, mode)
            rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
                   "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"])}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
                fut.result()
                if i % 50 == 0 or i == len(todo):
                    print(f"[g10] {mode} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="squad_200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
