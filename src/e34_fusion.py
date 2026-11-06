"""Fair test of "the negatives are an artefact of tuning for one direction".

Every mechanism was previously evaluated alone or bolted onto a stack that was
tuned for binding.  Here the signals are fused with weights fitted on a
*separate* dev split, then evaluated on the held-out eval split:

    score(r, d) = z(CE(r,d)) + w_sup * z(SUP(r,d)) + w_ans * GROUNDED(r,d)

If a joint configuration beats CE alone, the earlier negatives were confounded.
"""
import argparse, json, os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .e3_binding import em, f1, render
from .e4_pipeline import READ, fmt
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
PROMPT = ("Passage: {p}\nQuestion: {q}\n"
          "Answer with a short span copied verbatim from the passage. "
          "If the passage does not answer the question, output NONE.\nAnswer:")


def zs(v):
    mu = sum(v) / len(v)
    sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
    return [(x - mu) / sd for x in v]


class Scorer:
    def __init__(self, ce_path, sup_path):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(ce_path)
        self.ce = AutoModelForSequenceClassification.from_pretrained(ce_path).to(self.dev).eval()
        self.tok2 = AutoTokenizer.from_pretrained(sup_path)
        self.sup = AutoModelForSequenceClassification.from_pretrained(sup_path).to(self.dev).eval()

    def _run(self, m, t, q, docs, ml=256):
        out = []
        with torch.inference_mode():
            for i in range(0, len(docs), 64):
                c = docs[i:i + 64]
                b = t([q] * len(c), c, truncation=True, max_length=ml,
                      padding=True, return_tensors="pt").to(self.dev)
                out.extend(m(**b).logits.view(-1).tolist())
        return out

    def __call__(self, q, docs):
        return self._run(self.ce, self.tok, q, docs), self._run(self.sup, self.tok2, q, docs)


def load_split(path, lo, hi):
    return load(os.path.join(ROOT, "data", "raw", path))[lo:hi]


def main(ce_path, sup_path, tag="fusion", use_grounded=False, threads=8):
    sc = Scorer(ce_path, sup_path)
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def features(qs):
        out = []
        for q in qs:
            docs = [d["text"] for d in q["docs"]][:20]
            gold = {i for i, d in enumerate(q["docs"][:20]) if d["is_gold"]}
            reqs = [render(h["question"], [x["answer"] for x in q["hops"][:h["hop"]]])
                    for h in q["hops"][:2]]
            per = []
            for r in reqs:
                ce, su = sc(r or q["question"], docs)
                g = [0.0] * len(docs)
                if use_grounded:
                    for i, d in enumerate(docs):
                        try:
                            a = llm.chat([{"role": "user", "content":
                                           PROMPT.replace("{p}", d[:1200]).replace("{q}", r)}],
                                         max_tokens=24).strip().split("\n")[0].lower()
                        except Exception:
                            a = ""
                        g[i] = float(bool(a) and a != "none" and a in d.lower())
                per.append((zs(ce), zs(su), g))
            out.append({"qid": q["qid"], "question": q["question"], "gold": q["answer"], "gold_idx": gold,
                        "docs": docs, "per": per})
        return out

    def select(rec, w_sup, w_ans, k=2):
        sel = []
        for ce, su, g in rec["per"]:
            if len(sel) >= k:
                break
            rest = [i for i in range(len(rec["docs"])) if i not in sel]
            v = [ce[i] + w_sup * su[i] + w_ans * g[i] for i in rest]
            sel.append(rest[max(range(len(rest)), key=lambda t: v[t])])
        return sel

    dev = features(load_split("musique1000.jsonl", 200, 300))
    ev = features(load_split("musique200.jsonl", 0, 100))

    def cov(rs, w_sup, w_ans):
        return sum(len(r["gold_idx"] & set(select(r, w_sup, w_ans))) / max(len(r["gold_idx"]), 1)
                   for r in rs) / len(rs)

    grid = [0.0, 0.25, 0.5, 1.0, 2.0]
    best = max(((w, a) for w in grid for a in ([0.0] if not use_grounded else [0.0, 0.5, 1.0])),
               key=lambda t: cov(dev, t[0], t[1]))
    base_dev = cov(dev, 0.0, 0.0)
    print(json.dumps({"dev_base": base_dev, "dev_best": cov(dev, *best), "best_w": best,
                      "eval_base": cov(ev, 0.0, 0.0), "eval_best": cov(ev, *best)}, indent=2), flush=True)

    out = {"dev_base": base_dev, "dev_best": cov(dev, *best), "best_w": best,
           "eval_base": cov(ev, 0.0, 0.0), "eval_best": cov(ev, *best)}
    # reader EM on the eval split for the base and the fused selector
    recs = []
    for r in ev:
        for nm, w in (("base", (0.0, 0.0)), ("fused", best)):
            sel = select(r, *w)
            try:
                pred = llm.chat([{"role": "user", "content": READ.format(
                    q=r["question"], ps=fmt([r["docs"][i] for i in sel]))}], max_tokens=32).strip().split("\n")[0]
            except Exception:
                pred = ""
            recs.append({"qid": r["qid"], "mode": nm, "em": em(pred, r["gold"]), "f1": f1(pred, r["gold"])})
    per_q = [{"qid": r["qid"],
              "base": len(r["gold_idx"] & set(select(r, 0.0, 0.0))) / max(len(r["gold_idx"]), 1),
              "fused": len(r["gold_idx"] & set(select(r, *best))) / max(len(r["gold_idx"]), 1)}
             for r in ev]
    json.dump({"summary": out, "records": recs, "per_q": per_q},
              open(os.path.join(RES, f"e34_{tag}.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", required=True)
    ap.add_argument("--sup", required=True)
    ap.add_argument("--tag", default="fusion")
    ap.add_argument("--use-grounded", action="store_true")
    a = ap.parse_args()
    main(a.ce, a.sup, a.tag, a.use_grounded)
