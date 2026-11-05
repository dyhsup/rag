"""Answer-space matching: score a passage by whether the question can be
*groundedly answered from it*, instead of by topical similarity.

For every (requirement, candidate) pair we ask the generator for a short span
and keep the pair only if the span actually occurs in the passage.  The
resulting signal is binary and verifiable, and it is defined in answer space
rather than in text space.

Reported separately: the precision of the signal (does a grounded extraction
equal the hop's gold answer?), which bounds what any selector can do with it.
"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

import requests

from .ce_client import CERemote
from .e3_binding import em, f1, render
from .e4_pipeline import READ, fmt
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

PROMPT = ("Passage: {p}\nQuestion: {q}\n"
          "Answer with a short span copied verbatim from the passage. "
          "If the passage does not answer the question, output NONE.\nAnswer:")


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def main(n=100, k=2, workers=8, model=None, tag="answerspace"):
    qs = load()[:n]
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = None if model else CERemote()
    local = None
    if model:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(8)
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForSequenceClassification.from_pretrained(model).eval()

        def ce_rank(q, docs):
            out = []
            with torch.inference_mode():
                for i in range(0, len(docs), 32):
                    c = docs[i:i + 32]
                    b = tok([q] * len(c), c, truncation=True, max_length=256,
                            padding=True, return_tensors="pt")
                    out.extend(mdl(**b).logits.view(-1).tolist())
            return out
    else:
        def ce_rank(q, docs):
            return ce.rerank(q, docs)

    def extract(q, p):
        try:
            return llm.chat([{"role": "user", "content":
                              PROMPT.replace("{p}", p[:1200]).replace("{q}", q)}],
                            max_tokens=24).strip().split("\n")[0]
        except Exception:
            return ""

    def run_one(q):
        docs = [d["text"] for d in q["docs"]][:20]
        gold = {i for i, d in enumerate(q["docs"][:20]) if d["is_gold"]}
        hops = q["hops"][:2]
        reqs = [render(h["question"], [x["answer"] for x in q["hops"][:h["hop"]]]) for h in hops]
        hop_gold = [h["answer"] for h in hops]
        rows = []
        for ri, (req, ga) in enumerate(zip(reqs, hop_gold)):
            ex = []
            for i, d in enumerate(docs):
                a = extract(req, d)
                grounded = bool(norm(a)) and norm(a) != "none" and norm(a) in norm(d)
                ex.append({"i": i, "a": a, "grounded": grounded, "right": grounded and em(a, ga) > 0})
            rows.append(ex)
        # selector: grounded first, then CE for the requirement
        sel, diag = [], []
        for ri, req in enumerate(reqs):
            if len(sel) >= k:
                break
            rest = [i for i in range(len(docs)) if i not in sel]
            sc = ce_rank(req, [docs[i] for i in rest])
            order = sorted(range(len(rest)),
                           key=lambda t: (-int(rows[ri][rest[t]]["grounded"]), -sc[t]))
            sel.append(rest[order[0]])
        sel_cov = len(gold & set(sel)) / max(len(gold), 1)
        # diagnostics: precision/recall of the grounded signal
        gz = [(r, i) for ri in range(len(rows)) for r, i in
              [(e, e["i"]) for e in rows[ri]] if r["grounded"]]
        precision = (sum(1 for r, _ in gz if r["right"]) / len(gz)) if gz else 0.0
        # recall over *gold* pairs only: does the extractor ground the hop's
        # supporting paragraph?
        gp = hp = 0
        for ri, h in enumerate(hops):
            gi = next((i for i, d in enumerate(q["docs"][:20]) if d["idx"] == h["para_idx"]), None)
            if gi is None:
                continue
            hp += 1
            gp += int(rows[ri][gi]["grounded"])
        try:
            pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt([docs[i] for i in sel]))}],
                            max_tokens=32).strip().split("\n")[0]
        except Exception:
            pred = ""
        return {"qid": q["qid"], "gold": q["answer"], "cov": sel_cov, "sel": sel,
                "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"]),
                "signal_precision": precision, "signal_recall": (gp / hp) if hp else 0.0,
                "grounded_rate": len(gz) / (len(rows) * len(docs))}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(run_one, qs))

    out = {"n": len(recs), "k": k,
           "cov": sum(r["cov"] for r in recs) / len(recs),
           "EM": sum(r["em"] for r in recs) / len(recs),
           "F1": sum(r["f1"] for r in recs) / len(recs),
           "signal_precision": sum(r["signal_precision"] for r in recs) / len(recs),
           "signal_recall": sum(r["signal_recall"] for r in recs) / len(recs),
           "grounded_rate": sum(r["grounded_rate"] for r in recs) / len(recs)}
    print(json.dumps(out, indent=2), flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"e27_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "records": recs}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="answerspace")
    a = ap.parse_args()
    main(a.n, a.k, a.workers, a.model, a.tag)
