"""E3: automatic binding and error propagation.

Plan is oracle (MuSiQue decomposition). Only the binding is predicted, which
isolates error propagation from planning error.

Modes per question:
  none   : every hop queries with the original question (standard baseline)
  oracle : previous hop's GOLD answer substituted into the next requirement
  pred   : previous hop's EXTRACTED answer substituted (realistic)

Retrieval: BM25 over the pooled corpus. Reader: qwen3.6-27b, span-only prompt.
"""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data import tokenize
from .llm import LLM
from .musique import load
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_lock = threading.Lock()

READ_PROMPT = """Answer the question using only the passage. Output only the answer span, nothing else.

Question: {question}
Passage: {passage}
Answer:"""

_ARTICLES = {"a", "an", "the"}


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def em(pred, gold):
    return float(norm(pred) == norm(gold))


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(norm(pred) == norm(gold))
    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1
    hit = 0
    for w in g:
        if common.get(w, 0) > 0:
            hit += 1
            common[w] -= 1
    if hit == 0:
        return 0.0
    prec, rec = hit / len(p), hit / len(g)
    return 2 * prec * rec / (prec + rec)


def render(text, bindings):
    for i, ans in enumerate(bindings, 1):
        if ans:
            text = text.replace(f"#{i}", ans)
    text = re.sub(r"#\d+", " ", text)
    return text.replace(">>", " ").strip()


def read_answer(llm, question, passage):
    try:
        return llm.chat([{"role": "user", "content": READ_PROMPT.format(
            question=question, passage=passage[:2000])}], max_tokens=32).strip().split("\n")[0]
    except Exception:
        return ""


def run(n=100, topk=50, workers=8):
    qs = load()[:n]
    docs, keys = [], []
    for q in qs:
        for d in q["docs"]:
            docs.append({"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
                         "sentences": d["sentences"]})
            keys.append((d["title"], d["text"]))
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    out_path = os.path.join(RES, "e3_binding.jsonl")

    def one(q):
        gold_by_hop = {}
        for h in q["hops"]:
            gold_by_hop[h["hop"]] = h
        rows = {}
        for mode in ("none", "decomp", "oracle", "pred"):
            binds, hop_rows, margins = [], [], []
            for h in sorted(gold_by_hop):
                step = gold_by_hop[h]
                if mode == "none":
                    text = q["question"]
                elif mode == "decomp":
                    text = render(step["question"], [])
                else:
                    text = render(step["question"], binds)
                scores = list(index.score_docs(text))
                order = sorted(range(len(scores)), key=lambda i: -scores[i])[:topk]
                gold_doc = next((d for d in q["docs"] if d["idx"] == step["para_idx"]), None)
                rank = -1
                if gold_doc is not None:
                    gkey = (gold_doc["title"], gold_doc["text"])
                    if gkey in keys:
                        gi = keys.index(gkey)
                        rank = sum(1 for x in scores if x > scores[gi]) + 1
                margin = (scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
                # reader over the retrieved top-1
                pred = read_answer(llm, text, docs[order[0]]["text"])
                if mode == "oracle":
                    binds.append(step["answer"])
                elif mode == "pred":
                    binds.append(pred)
                hop_rows.append({
                    "hop": h, "query": text, "rank": rank,
                    "pred": pred, "gold": step["answer"],
                    "em": em(pred, step["answer"]), "f1": f1(pred, step["answer"]),
                    "margin": margin,
                })
                margins.append(margin)
            rows[mode] = hop_rows
        e2e = {}
        for m in ("none", "decomp", "oracle", "pred"):
            last = rows[m][-1]
            e2e[m] = {"pred": last["pred"], "em": em(last["pred"], q["answer"]),
                      "f1": f1(last["pred"], q["answer"])}
        rec = {"qid": q["qid"], "n_hops": q["n_hops"], "gold_answer": q["answer"],
               "e2e": e2e, "modes": rows}
        with _lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rec

    # resume
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["qid"] for l in f if l.strip()}
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[e3] {len(todo)} questions to run ({len(done)} cached)")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, q) for q in todo]), 1):
            fut.result()
            if i % 10 == 0 or i == len(todo):
                print(f"[e3] {i}/{len(todo)} calls={llm.calls} hits={llm.cache_hits}", flush=True)

    summarize(out_path)


def summarize(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    stats = {}
    for mode in ("none", "decomp", "oracle", "pred"):
        by_hop = {}
        for r in recs:
            for h in r["modes"][mode]:
                by_hop.setdefault(h["hop"], []).append(h)
        stats[mode] = {
            "recall@50": {p: sum(1 for h in v if 0 < h["rank"] <= 50) / len(v) for p, v in by_hop.items()},
            "hop_em": {p: sum(h["em"] for h in v) / len(v) for p, v in by_hop.items()},
            "n": {p: len(v) for p, v in by_hop.items()},
        }
    e2e = {m: {"em": sum(r["e2e"][m]["em"] for r in recs) / len(recs),
               "f1": sum(r["e2e"][m]["f1"] for r in recs) / len(recs)}
           for m in ("none", "decomp", "oracle", "pred")}
    lines = ["# E3 automatic binding: error propagation (MuSiQue)", "",
             f"- questions: {len(recs)}", "",
             "## retrieval recall@50 by hop and binding mode", "",
             "| hop | none | decomp only | oracle binding | predicted binding |", "|---|---|---|---|---|"]
    hops = sorted(stats["none"]["recall@50"])
    for p in hops:
        lines.append(f"| {p} | {stats['none']['recall@50'][p]:.3f} | {stats['decomp']['recall@50'][p]:.3f} | "
                     f"{stats['oracle']['recall@50'][p]:.3f} | {stats['pred']['recall@50'][p]:.3f} |")
    lines += ["", "## hop answer EM (reader quality + binding)", "",
              "| hop | decomp only | oracle binding | predicted binding |", "|---|---|---|---|"]
    for p in hops:
        lines.append(f"| {p} | {stats['decomp']['hop_em'][p]:.3f} | {stats['oracle']['hop_em'][p]:.3f} | "
                     f"{stats['pred']['hop_em'][p]:.3f} |")
    lines += ["", "## end-to-end (final answer vs gold)", "",
              "| mode | EM | F1 |", "|---|---|---|"]
    for m in ("none", "decomp", "oracle", "pred"):
        lines.append(f"| {m} | {e2e[m]['em']:.3f} | {e2e[m]['f1']:.3f} |")
    lines.append("")
    with open(os.path.join(RES, "e3_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "e3_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "e2e": e2e, "n": len(recs)}, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run(a.n, a.workers)
