"""Evaluate set-level matching: score every candidate *pair* and take the best.

This is a prototype of the idea (set-level, not pointwise); exhaustive pair
enumeration is fine at k=2 and would need beam search at larger k.
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
MAXD = 450


def main(n=100, model=None, ce_model=None, threads=8, tag="setmatch"):
    qs = load()[:n]
    torch.set_num_threads(threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model).to(dev).eval()
    ce = CERemote() if ce_model else None
    if ce_model:
        tok2 = AutoTokenizer.from_pretrained(ce_model)
        mdl2 = AutoModelForSequenceClassification.from_pretrained(ce_model).to(dev).eval()
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def score_pairs(q, docs, pairs):
        a = [q] * len(pairs)
        b = [docs[i][:MAXD] + " || " + docs[j][:MAXD] for i, j in pairs]
        out = []
        with torch.inference_mode():
            for s in range(0, len(a), 64):
                bb = tok(a[s:s + 64], b[s:s + 64], truncation=True, max_length=384,
                         padding=True, return_tensors="pt").to(dev)
                out.extend(mdl(**bb).logits.view(-1).tolist())
        return out

    def ce_rank(q, docs):
        if not ce_model:
            return ce.rerank(q, docs)
        out = []
        with torch.inference_mode():
            for i in range(0, len(docs), 64):
                c = docs[i:i + 64]
                bb = tok2([q] * len(c), c, truncation=True, max_length=256,
                          padding=True, return_tensors="pt").to(dev)
                out.extend(mdl2(**bb).logits.view(-1).tolist())
        return out

    recs = []
    for q in qs:
        docs = [d["text"] for d in q["docs"]][:20]
        gold = {i for i, d in enumerate(q["docs"][:20]) if d["is_gold"]}
        pairs = [(i, j) for i in range(len(docs)) for j in range(i + 1, len(docs))]
        sc = score_pairs(q["question"], docs, pairs)
        sel = list(pairs[max(range(len(pairs)), key=lambda t: sc[t])])
        # per-requirement CE selection on the same pool, for the paired comparison
        reqs = [render(h["question"], [x["answer"] for x in q["hops"][:h["hop"]]]) for h in q["hops"][:4]]
        sel_ce = []
        for r in reqs[:2]:
            rest = [i for i in range(len(docs)) if i not in sel_ce]
            s2 = ce_rank(r, [docs[i] for i in rest])
            sel_ce.append(rest[max(range(len(rest)), key=lambda t: s2[t])])
        rec = {"qid": q["qid"], "gold": q["answer"],
               "set_cov": len(gold & set(sel)) / max(len(gold), 1),
               "ce_cov": len(gold & set(sel_ce)) / max(len(gold), 1),
               "set_sel": sel, "ce_sel": sel_ce}
        for nm, s in (("set", sel), ("ce", sel_ce)):
            try:
                pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt([docs[i] for i in s]))}],
                                max_tokens=32).strip().split("\n")[0]
            except Exception:
                pred = ""
            rec[f"{nm}_em"] = em(pred, q["answer"])
            rec[f"{nm}_f1"] = f1(pred, q["answer"])
        recs.append(rec)
    out = {"n": len(recs)}
    for m in ("set", "ce"):
        out[f"{m}_cov"] = sum(r[f"{m}_cov"] for r in recs) / len(recs)
        out[f"{m}_EM"] = sum(r[f"{m}_em"] for r in recs) / len(recs)
        out[f"{m}_F1"] = sum(r[f"{m}_f1"] for r in recs) / len(recs)
    print(json.dumps(out, indent=2), flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"e28_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "records": recs}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ce-model", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="setmatch")
    a = ap.parse_args()
    main(a.n, a.model, a.ce_model, a.threads, a.tag)
