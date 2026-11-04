"""E0-core: redundancy blindness of pointwise top-k.

Pool per question: gold1, gold2, meaning-preserving rewrites of each (which are
valid evidence for the same requirement), plus irrelevant negatives. Grounding
labels are oracle, so any difference between selectors is due to the selection
objective, not to matcher quality.
"""
import argparse
import json
import os

from . import eval as ev
from .data import load_questions
from .probe import read_jsonl, split_sents, RES
from .retriever import QuestionIndex

Ks = [1, 2, 3, 4, 5, 6]


def build_pool(q, variants_by_doc, n_neg=6):
    gold = [d for d in q["docs"] if d["is_gold"]]
    docs = []
    for gi, d in enumerate(gold):
        docs.append({
            "doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
            "sentences": d["sentences"], "kind": "orig", "covers": {gi},
        })
        for v in variants_by_doc.get(d["doc_id"], []):
            docs.append({
                "doc_id": f'{d["doc_id"]}#{v["style"]}', "title": d["title"],
                "text": v["text"], "sentences": split_sents(v["text"]),
                "kind": v["style"], "covers": {gi},
            })
    negs = [d for d in q["docs"] if not d["is_gold"]][:n_neg]
    for d in negs:
        docs.append({
            "doc_id": f'{q["qid"]}::neg::{d["title"]}', "title": d["title"],
            "text": d["text"], "sentences": d["sentences"], "kind": "neg", "covers": set(),
        })
    return docs, len(gold)


def select_pointwise(scores, k):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def select_coverage(scores, covers, k):
    chosen, covered = [], set()
    for _ in range(k):
        best, best_key = None, None
        for i in range(len(scores)):
            if i in chosen:
                continue
            key = (len(covers[i] - covered), scores[i])
            if best_key is None or key > best_key:
                best, best_key = i, key
        if best is None:
            break
        chosen.append(best)
        covered |= covers[best]
    return chosen


def run(n_questions=10):
    questions = load_questions()
    variants = [v for v in read_jsonl(os.path.join(RES, "variants.jsonl"))
                if v.get("equivalent") and v.get("text")]
    by_q = {}
    for v in variants:
        by_q.setdefault(v["qid"], {}).setdefault(v["doc_id"], []).append(v)

    curve = {k: {"pointwise": [], "coverage": []} for k in Ks}
    rows = []
    for q in questions[:n_questions]:
        vb = by_q.get(q["qid"], {})
        if not vb:
            continue
        pool, n_gold = build_pool(q, vb)
        idx = QuestionIndex({"qid": q["qid"], "docs": pool})
        scores = idx.score_docs(q["question"])
        covers = [d["covers"] for d in pool]

        row = {"qid": q["qid"], "question": q["question"], "n_pool": len(pool),
               "n_gold": n_gold, "per_k": {}}
        for k in Ks:
            pw = select_pointwise(scores, k)
            cv = select_coverage(scores, covers, k)
            cov_pw = len(set().union(*[covers[i] for i in pw])) / n_gold
            cov_cv = len(set().union(*[covers[i] for i in cv])) / n_gold
            curve[k]["pointwise"].append(cov_pw)
            curve[k]["coverage"].append(cov_cv)
            row["per_k"][k] = {
                "pointwise": cov_pw, "coverage": cov_cv,
                "pointwise_ids": [pool[i]["doc_id"] for i in pw],
                "coverage_ids": [pool[i]["doc_id"] for i in cv],
            }
        rows.append(row)

    summary = {}
    for k in Ks:
        summary[k] = {
            "pointwise": {"mean": ev.mean(curve[k]["pointwise"]),
                          "ci": ev.bootstrap_ci(curve[k]["pointwise"])},
            "coverage": {"mean": ev.mean(curve[k]["coverage"]),
                         "ci": ev.bootstrap_ci(curve[k]["coverage"])},
        }

    out = {"n_questions": len(rows), "summary": summary, "per_question": rows}
    with open(os.path.join(RES, "e0_redundancy.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    report(out)
    return out


def report(out):
    lines = ["# E0-core: pointwise top-k is redundancy-blind (oracle grounding labels)", ""]
    lines.append(f"- questions: {out['n_questions']}　|　pool = 2 gold + their equivalent rewrites + 6 negatives")
    lines.append("")
    lines.append("| k | pointwise coverage | greedy coverage | gap |")
    lines.append("|---|---|---|---|")
    for k, d in out["summary"].items():
        p = d["pointwise"]["mean"]
        c = d["coverage"]["mean"]
        lines.append(f"| {k} | {p:.3f} | {c:.3f} | {c - p:+.3f} |")
    lines.append("")
    k2 = out["summary"].get(2)
    if k2:
        lines.append(f"At k=2: pointwise {k2['pointwise']['mean']:.3f} vs coverage "
                     f"{k2['coverage']['mean']:.3f}")
        lines.append("")
    lines.append("## per-question at k=2")
    lines.append("")
    lines.append("| question | pointwise picks | coverage picks |")
    lines.append("|---|---|---|")
    for r in out["per_question"]:
        if 2 not in r["per_k"]:
            continue
        pw = [x.split("#")[-1] if "#" in x else ("NEG" if "::neg::" in x else "orig")
              for x in r["per_k"][2]["pointwise_ids"]]
        cv = [x.split("#")[-1] if "#" in x else ("NEG" if "::neg::" in x else "orig")
              for x in r["per_k"][2]["coverage_ids"]]
        lines.append(f"| {r['question'][:60]} | {pw} | {cv} |")
    path = os.path.join(RES, "e0_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[e0] wrote {path}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    run(ap.parse_args().n)
