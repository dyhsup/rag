"""Does a second query channel rescue the hops where hard binding fails?

Realistic regime: requirements and bindings come from the *pipeline*
(LLM plan + predicted answers), not from gold.  For each hop we compare
  A bound       : plan hop with the predicted previous answers substituted
  B descriptor  : plan hop with the previous plan *queries* substituted
  C raw         : no substitution
and report, among the hops where A misses the gold paragraph in top-k, how
often B or C still finds it.  That rescue rate is the headroom of any hedge.
"""
import argparse, json, os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .e3_binding import render
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def main(n=200, model=None, k=3, trace=None, plans=None, tag="rescue"):
    qs = {q["qid"]: q for q in load()[:n]}
    plans = {json.loads(l)["qid"]: json.loads(l)["hops"] for l in open(plans, encoding="utf-8")}
    tr = {}
    for line in open(trace, encoding="utf-8"):
        r = json.loads(line)
        tr[r["qid"]] = r["trace"]
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
    for qid, hops in plans.items():
        if qid not in tr or qid not in qs:
            continue
        q = qs[qid]
        docs = [d["text"] for d in q["docs"][:20]]
        t = tr[qid]
        for hi, hop in enumerate(hops[:4]):
            if hi >= len(t):
                break
            hq = hop["query"] if isinstance(hop, dict) else hop
            gi = None
            # map the plan hop back to the dataset hop by order
            if hi < len(q["hops"]):
                gi = next((i for i, d in enumerate(q["docs"][:20])
                           if d["idx"] == q["hops"][hi]["para_idx"]), None)
            if gi is None:
                continue
            preds = [t[j]["answer"] for j in range(hi)]
            prevq = [t[j]["query"] for j in range(hi)]
            ch = {"A": render(hq, preds), "B": render(hq, prevq), "C": render(hq, [])}
            rec = {"qid": qid, "hop": hi, "gold": gi}
            sc = {c: rank(v or q["question"], docs) for c, v in ch.items()}
            for c in "ABC":
                order = sorted(range(len(docs)), key=lambda i: -sc[c][i])[:k]
                rec[f"{c}_r"] = int(gi in order)
            rows.append(rec)

    miss = [r for r in rows if not r["A_r"]]
    out = {"n_hops": len(rows), "k": k,
           "A_recall": sum(r["A_r"] for r in rows) / len(rows),
           "B_recall": sum(r["B_r"] for r in rows) / len(rows),
           "C_recall": sum(r["C_r"] for r in rows) / len(rows),
           "A_misses": len(miss),
           "rescue_by_B": sum(r["B_r"] for r in miss) / len(miss) if miss else 0.0,
           "rescue_by_C": sum(r["C_r"] for r in miss) / len(miss) if miss else 0.0,
           "rescue_by_BC": sum(int(r["B_r"] or r["C_r"]) for r in miss) / len(miss) if miss else 0.0}
    print(json.dumps(out, indent=2), flush=True)
    with open(os.path.join(RES, f"e33_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "records": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--trace", default="/root/reqce/results/e13_select_musique200_gpu.jsonl")
    ap.add_argument("--plans", default="/root/reqce/results/e13_plans_musique200.jsonl")
    ap.add_argument("--tag", default="rescue")
    a = ap.parse_args()
    main(a.n, a.model, a.k, a.trace, a.plans, a.tag)
