"""Requirement-conditioned reranking: does binding the query at the scoring
layer close the query-evidence gap?

Selection modes
  whole : score every passage against the full question, take top-k
  seq   : for each requirement in order, take its best passage from the rest
  global: score passage i by max_r score(r, passage i), take top-k

Requirement sources
  gold  : MuSiQue decomposition, hop queries bound with the gold answers
  llm   : the router's plan, hop queries bound with the predicted answers
"""
import argparse, json, os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .ce_client import CERemote
from .e3_binding import em, f1, render
from .e4_pipeline import READ, fmt
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


class LocalCE:
    def __init__(self, path, threads=4, ml=192):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.ml = ml

    def rerank(self, q, docs, bs=32):
        out = []
        with torch.inference_mode():
            for i in range(0, len(docs), bs):
                c = docs[i:i + bs]
                b = self.tok([q] * len(c), c, truncation=True, max_length=self.ml,
                             padding=True, return_tensors="pt")
                out.extend(self.m(**b).logits.view(-1).tolist())
        return out


def select(mode, rerank, reqs, docs, k):
    if mode == "whole":
        s = rerank(reqs[0], docs)
        return sorted(range(len(docs)), key=lambda i: -s[i])[:k]
    per = [rerank(r, docs) for r in reqs]
    if mode == "seq":
        sel = []
        for s in per:
            if len(sel) >= k:
                break
            rest = [i for i in range(len(docs)) if i not in sel]
            sel.append(max(rest, key=lambda i: s[i]))
        return sel
    if mode == "cover":
        # submodular requirement coverage: greedily maximise the gain in
        # requirement coverage, normalised per requirement so that scores from
        # different requirements are comparable.
        norm = []
        for p in per:
            lo, hi = min(p), max(p)
            norm.append([(v - lo) / (hi - lo) if hi > lo else 0.0 for v in p])
        cov = [max(n) for n in norm]          # best achievable per requirement
        cur = [0.0] * len(norm)
        sel = []
        while len(sel) < k:
            rest = [i for i in range(len(docs)) if i not in sel]
            best_i, best_g = None, -1e9
            for i in rest:
                gain = sum(max(0.0, norm[j][i] - cur[j]) for j in range(len(norm)))
                if gain > best_g:
                    best_g, best_i = gain, i
            if best_i is None:
                break
            sel.append(best_i)
            for j in range(len(norm)):
                cur[j] = max(cur[j], norm[j][best_i])
        return sel
    if mode == "global":
        best = [max(p[i] for p in per) for i in range(len(docs))]
        return sorted(range(len(docs)), key=lambda i: -best[i])[:k]
    raise ValueError(mode)


def main(n=100, k=2, model=None, modes=("whole", "seq", "global"), plan="gold",
         threads=4, read=False, tag="reqce"):
    from concurrent.futures import ThreadPoolExecutor
    qs = load()[:n]
    rr = LocalCE(model, threads) if model else None
    remote = None if model else CERemote()
    rerank = rr.rerank if rr else (lambda q, d: remote.rerank(q, d))
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def reqs_for(q):
        if plan == "gold":
            return [render(h["question"], [x["answer"] for x in q["hops"][:h["hop"]]]) or q["question"]
                    for h in q["hops"][:4]]
        from .e14_router import plan as make_plan
        p = make_plan(llm, q)
        return [h["query"].replace("#1", "").strip() or q["question"] for h in p["hops"]] or [q["question"]]

    def run_one(q):
        docs = [d["text"] for d in q["docs"]][:20]
        gold = {i for i, d in enumerate(q["docs"][:20]) if d["is_gold"]}
        reqs = reqs_for(q)
        rec = {"qid": q["qid"], "n_req": len(reqs), "reqs": reqs, "gold": q["answer"]}
        for mode in modes:
            sel = select(mode, rerank, reqs if mode != "whole" else [q["question"]], docs, k)
            rec[f"{mode}_cov"] = len(gold & set(sel)) / max(len(gold), 1)
            rec[f"{mode}_any"] = int(bool(gold & set(sel)))
            rec[f"{mode}_sel"] = sel
            if read:
                try:
                    pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt([docs[i] for i in sel]))}],
                                    max_tokens=32).strip().split("\n")[0]
                except Exception:
                    pred = ""
                rec[f"{mode}_em"] = em(pred, q["answer"])
                rec[f"{mode}_f1"] = f1(pred, q["answer"])
        return rec

    with ThreadPoolExecutor(max_workers=threads) as ex:
        recs = list(ex.map(run_one, qs))

    summary = {"n": len(recs), "k": k, "plan": plan, "model": model or "remote"}
    for mode in modes:
        summary[f"{mode}_cov"] = sum(r[f"{mode}_cov"] for r in recs) / len(recs)
        summary[f"{mode}_any"] = sum(r[f"{mode}_any"] for r in recs) / len(recs)
        if read:
            summary[f"{mode}_EM"] = sum(r[f"{mode}_em"] for r in recs) / len(recs)
            summary[f"{mode}_F1"] = sum(r[f"{mode}_f1"] for r in recs) / len(recs)
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"e23_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": recs}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--model", default=None)
    ap.add_argument("--modes", default="whole,seq,global")
    ap.add_argument("--plan", default="gold", choices=["gold", "llm"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--tag", default="reqce")
    a = ap.parse_args()
    main(a.n, a.k, a.model, tuple(a.modes.split(",")), a.plan, a.threads, a.read, a.tag)
