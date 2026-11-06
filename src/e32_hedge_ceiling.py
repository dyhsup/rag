"""Is there headroom for hedging the query formulation?

Three query channels for the same requirement:
  A bound       : substitute the previous *answer*            (current method)
  B descriptor  : substitute the previous *sub-question*      (a definite description)
  C raw         : no substitution at all

Channel B is the interesting one: it stays correct as long as the previous
*question* is right, even if the previous *answer* was wrong, so it does not
need online binding-error detection.

We measure per-hop recall of the gold paragraph for each channel, and for the
union of channels, which upper-bounds any fusion rule.
"""
import argparse, json, os
import statistics

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .e3_binding import render
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def main(n=100, model=None, k=3, tag="hedge"):
    qs = load()[:n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model).to(dev).eval()

    def rank(q, docs):
        out = []
        with torch.inference_mode():
            for i in range(0, len(docs), 64):
                c = docs[i:i + 64]
                b = tok([q] * len(c), c, truncation=True, max_length=256,
                        padding=True, return_tensors="pt").to(dev)
                out.extend(mdl(**b).logits.view(-1).tolist())
        return out

    rows = []
    for q in qs:
        docs = [d["text"] for d in q["docs"]][:20]
        for hi, h in enumerate(q["hops"][:4]):
            gi = next((i for i, d in enumerate(q["docs"][:20]) if d["idx"] == h["para_idx"]), None)
            if gi is None:
                continue
            prev_ans = [x["answer"] for x in q["hops"][:hi]]
            prev_q = [x["question"] for x in q["hops"][:hi]]
            ch = {"A": render(h["question"], prev_ans),
                  "B": render(h["question"], prev_q),
                  "C": render(h["question"], [])}
            sc = {c: rank(v or q["question"], docs) for c, v in ch.items()}
            rec = {"qid": q["qid"], "hop": hi, "gold": gi, "queries": ch}
            for c in "ABC":
                order = sorted(range(len(docs)), key=lambda i: -sc[c][i])[:k]
                rec[f"{c}_r"] = int(gi in order)
                rec[f"{c}_rank"] = order.index(gi) if gi in order else 99
            best = {}
            for c in "ABC":
                for r_, i in enumerate(sorted(range(len(docs)), key=lambda i: -sc[c][i])):
                    best.setdefault(i, 1e9)
                    best[i] = min(best[i], r_)
            union = sorted(best, key=lambda i: best[i])[:k]
            rec["U_r"] = int(gi in union)
            for pair in ("AB", "AC", "ABC"):
                b2 = {}
                for c in pair:
                    for r_, i in enumerate(sorted(range(len(docs)), key=lambda i: -sc[c][i])):
                        b2.setdefault(i, 1e9); b2[i] = min(b2[i], r_)
                top = sorted(b2, key=lambda i: b2[i])[:k]
                rec[f"{pair}_r"] = int(gi in top)
            rows.append(rec)

    out = {"n_hops": len(rows), "k": k}
    for c in ("A", "B", "C", "U", "AB", "AC", "ABC"):
        out[f"recall@{k}_{c}"] = sum(r[f"{c}_r"] for r in rows) / len(rows)
    print(json.dumps(out, indent=2), flush=True)
    with open(os.path.join(RES, f"e32_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "records": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tag", default="hedge")
    a = ap.parse_args()
    main(a.n, a.model, a.k, a.tag)
