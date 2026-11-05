"""Is the query-slot effect dataset-dependent because coverage is already saturated?

Runs the E23 whole vs per-hop comparison on 2WikiMultihopQA, where the
end-to-end gain was null.
"""
import argparse, json, os, statistics, sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .ce_client import CERemote
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(n=100, model=None, tag="wiki2"):
    torch.set_num_threads(4)
    qs = load(os.path.join(ROOT, "data", "raw", "wiki2_200.jsonl"))[:n]
    if model:
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForSequenceClassification.from_pretrained(model).eval()

        def rerank(q, docs):
            out = []
            with torch.inference_mode():
                for i in range(0, len(docs), 32):
                    c = docs[i:i + 32]
                    b = tok([q] * len(c), c, truncation=True, max_length=192,
                            padding=True, return_tensors="pt")
                    out.extend(mdl(**b).logits.view(-1).tolist())
            return out
    else:
        ce = CERemote()

        def rerank(q, docs):
            return ce.rerank(q, docs)

    rows = []
    for q in qs:
        docs = [d["text"] for d in q["docs"]]
        gold = {i for i, d in enumerate(q["docs"]) if d["is_gold"]}
        if not gold:
            continue
        s = rerank(q["question"], docs)
        whole = sorted(range(len(docs)), key=lambda i: -s[i])[:2]
        sel = []
        for h in q["hops"][:2]:
            s2 = rerank(h["question"], docs)
            rest = [i for i in range(len(docs)) if i not in sel]
            sel.append(max(rest, key=lambda i: s2[i]))
        rows.append({"qid": q["qid"], "n_gold": len(gold),
                     "whole_cov": len(gold & set(whole)) / len(gold),
                     "seq_cov": len(gold & set(sel)) / len(gold),
                     "whole_any": int(bool(gold & set(whole))),
                     "seq_any": int(bool(gold & set(sel)))})
    out = {"n": len(rows), "model": model or "remote"}
    for m in ("whole", "seq"):
        out[f"{m}_cov"] = statistics.mean(r[f"{m}_cov"] for r in rows)
        out[f"{m}_any"] = statistics.mean(r[f"{m}_any"] for r in rows)
    print(json.dumps(out, indent=2))
    json.dump({"summary": out, "records": rows},
              open(os.path.join(ROOT, "results", f"e25_{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="wiki2")
    a = ap.parse_args()
    main(a.n, a.model, a.tag)
