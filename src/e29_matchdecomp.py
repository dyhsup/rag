"""Decompose for *matchability* rather than for reasoning.

Two planners produce candidate decompositions:
  A (reasoning-first): the original router prompt.
  B (matchability-first): "split into the fewest independently answerable
    sub-questions, each naming at most one entity".
Each candidate is scored by how well its requirements can be matched by the
pool, avg_i max_d CE(r_i, d), and the winner is executed.
"""
import argparse, json, os, statistics

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .e3_binding import em, f1, render
from .e4_pipeline import READ, fmt
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

PLAN_A = """Decide whether the question requires chaining facts across entities.
If multi-hop, split it into 2-4 hop queries. Use "#1" to refer to the answer of hop 1.
Output JSON only: {{"multi_hop": true/false, "hops": [{{"query": "..."}}]}}

Question: {q}
Output:"""

PLAN_B = """Split the question into the fewest sub-questions that can each be answered
from a SINGLE encyclopedia paragraph, and that each name at most one entity.
Do not write reasoning steps; write only retrievable sub-questions.
Use "#1" to refer to the answer of sub-question 1.
Output JSON only: {{"multi_hop": true/false, "hops": [{{"query": "..."}}]}}

Question: {q}
Output:"""


def hops_of(llm, prompt, q):
    try:
        obj = llm.json_call(prompt.replace("{q}", q["question"]), max_tokens=320)
        hs = [h["query"] for h in obj.get("hops", []) if h.get("query")][:4]
        return hs or [q["question"]]
    except Exception:
        return [q["question"]]


def main(n=100, model=None, threads=8, tag="matchdecomp"):
    qs = load()[:n]
    torch.set_num_threads(threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model).to(dev).eval()
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def rank(q, docs):
        out = []
        with torch.inference_mode():
            for i in range(0, len(docs), 64):
                c = docs[i:i + 64]
                b = tok([q] * len(c), c, truncation=True, max_length=256,
                        padding=True, return_tensors="pt").to(dev)
                out.extend(mdl(**b).logits.view(-1).tolist())
        return out

    def matchability(reqs, docs):
        return statistics.mean(max(rank(r, docs)) for r in reqs)

    def execute(reqs, docs, k=2):
        sel = []
        for r in reqs:
            if len(sel) >= k:
                break
            rest = [i for i in range(len(docs)) if i not in sel]
            s = rank(r, [docs[i] for i in rest])
            sel.append(rest[max(range(len(rest)), key=lambda t: s[t])])
        return sel

    recs = []
    for q in qs:
        docs = [d["text"] for d in q["docs"]][:20]
        gold = {i for i, d in enumerate(q["docs"][:20]) if d["is_gold"]}
        pa = hops_of(llm, PLAN_A, q)
        pb = hops_of(llm, PLAN_B, q)
        cands = {"A": pa, "B": pb}
        scores = {"A": matchability(pa, docs)}
        if pb != pa:
            scores["B"] = matchability(pb, docs)
        pick = max(list(cands), key=lambda k: scores.get(k, -1e9))
        rec = {"qid": q["qid"], "gold": q["answer"], "plans": cands, "scores": scores, "pick": pick}
        for nm, reqs in cands.items():
            sel = execute(reqs, docs)
            cov = len(gold & set(sel)) / max(len(gold), 1)
            try:
                pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt([docs[i] for i in sel]))}],
                                max_tokens=32).strip().split("\n")[0]
            except Exception:
                pred = ""
            rec[f"{nm}_cov"] = cov
            rec[f"{nm}_em"] = em(pred, q["answer"])
            rec[f"{nm}_f1"] = f1(pred, q["answer"])
        rec["same"] = int(pb == pa)
        recs.append(rec)
    out = {"n": len(recs), "same_plan_rate": sum(r["same"] for r in recs) / len(recs),
           "picked_B": sum(r["pick"] == "B" for r in recs) / len(recs)}
    for m in ("A", "B"):
        out[f"{m}_cov"] = sum(r[f"{m}_cov"] for r in recs) / len(recs)
        out[f"{m}_EM"] = sum(r[f"{m}_em"] for r in recs) / len(recs)
    print(json.dumps(out, indent=2), flush=True)
    with open(os.path.join(RES, f"e29_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "records": recs}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="matchdecomp")
    a = ap.parse_args()
    main(a.n, a.model, a.threads, a.tag)
