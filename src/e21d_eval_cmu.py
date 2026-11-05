"""Evaluate the distilled CMU scorer against CE pointwise selection.

Both selectors use the same 20-passage pool, the same k, and the same reader.
Metrics: gold-paragraph recall in the selected set, EM and F1 after reading.
"""
import argparse, json, os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .ce_client import CERemote
from .cmu_fmt import text_a, text_b
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def main(n=100, k=2, model_dir=None, tag="cmu", workers=8, skip_read=False):
    from concurrent.futures import ThreadPoolExecutor
    model_dir = model_dir or os.path.join(ROOT, "dist", "cmu", "model")
    qs = load()[:n]
    torch.set_num_threads(workers)
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    ce = CERemote()
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def score_batch(q, sel, cands):
        a = text_a(q, sel)
        b = tok([a] * len(cands), [text_b(c) for c in cands],
                truncation=True, max_length=320, padding=True, return_tensors="pt")
        with torch.inference_mode():
            return mdl(**b).logits.view(-1).tolist()

    def read(q, ps):
        try:
            return llm.chat([{"role": "user", "content": READ.format(q=q, ps=fmt(ps))}],
                            max_tokens=32).strip().split("\n")[0]
        except Exception:
            return ""

    def run_one(q):
        docs = [d["text"] for d in q["docs"]][:20]
        gold = {i for i, d in enumerate(q["docs"][:20]) if d["is_gold"]}
        sel, sel_ce = [], []
        csc = ce.rerank(q["question"], docs)
        sel_ce = sorted(range(len(docs)), key=lambda i: -csc[i])[:k]
        for _ in range(k):
            rest = [i for i in range(len(docs)) if i not in sel]
            if not rest:
                break
            vals = score_batch(q["question"], [docs[j] for j in sel], [docs[i] for i in rest])
            sel.append(rest[max(range(len(rest)), key=lambda t: vals[t])])
        out = {"qid": q["qid"], "gold": q["answer"],
               "cmu_recall": len(gold & set(sel)) / max(len(gold), 1),
               "ce_recall": len(gold & set(sel_ce)) / max(len(gold), 1),
               "cmu_sel": sel, "ce_sel": sel_ce}
        if not skip_read:
            for name, s in (("cmu", sel), ("ce", sel_ce)):
                pred = read(q["question"], [docs[i] for i in s])
                out[f"{name}_pred"] = pred
                out[f"{name}_em"] = em(pred, q["answer"])
                out[f"{name}_f1"] = f1(pred, q["answer"])
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(run_one, qs))

    summary = {"n": len(recs), "k": k,
               "cmu_recall": sum(r["cmu_recall"] for r in recs) / len(recs),
               "ce_recall": sum(r["ce_recall"] for r in recs) / len(recs)}
    if not skip_read:
        for m in ("cmu", "ce"):
            summary[f"{m}_EM"] = sum(r[f"{m}_em"] for r in recs) / len(recs)
            summary[f"{m}_F1"] = sum(r[f"{m}_f1"] for r in recs) / len(recs)
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"e21d_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": recs}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="cmu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-read", action="store_true")
    a = ap.parse_args()
    main(a.n, a.k, a.model, a.tag, a.workers, a.skip_read)
