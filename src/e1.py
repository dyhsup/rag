"""E1: does the coverage objective still matter on natural corpora?

Two settings:
  1. per-question pool (19-20 MuSiQue paragraphs) - budget allocation across hops
  2. pooled corpus (~4000 paragraphs from 200 questions) - natural duplication,
     candidates restricted to BM25 top-50 like a real retriever

Grounding labels are oracle (MuSiQue question_decomposition), so differences are
due to the selection objective only.
"""
import argparse
import json
import os

from . import eval as ev
from .musique import load
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
KS = [1, 2, 3, 4, 6]


def hop_gold(q):
    """hop -> (title, text) of its supporting paragraph."""
    by_idx = {d["idx"]: d for d in q["docs"]}
    out = {}
    for h in q["hops"]:
        d = by_idx.get(h["para_idx"])
        if d:
            out[h["hop"]] = (d["title"], d["text"])
    return out


def pointwise(scores, k):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def greedy(scores, covers, k):
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


def per_question(questions, ks=KS):
    curve = {k: {"pointwise": [], "coverage": [], "redundancy": []} for k in ks}
    for q in questions:
        idx = QuestionIndex({"qid": q["qid"], "docs": q["docs"]})
        scores = idx.score_docs(q["question"])
        gold = hop_gold(q)
        key2hop = {v: h for h, v in gold.items()}
        covers = []
        for d in q["docs"]:
            h = key2hop.get((d["title"], d["text"]))
            covers.append({h} if h is not None else set())
        n_hops = len(gold)

        for k in ks:
            pw = pointwise(scores, k)
            cv = greedy(scores, covers, k)
            cov_pw = len(set().union(*[covers[i] for i in pw])) / n_hops
            cov_cv = len(set().union(*[covers[i] for i in cv])) / n_hops
            seen, red = set(), 0
            for i in pw:
                if covers[i] and covers[i] <= seen:
                    red += 1
                seen |= covers[i]
            curve[k]["pointwise"].append(cov_pw)
            curve[k]["coverage"].append(cov_cv)
            curve[k]["redundancy"].append(red / max(1, len(pw)))
    return curve


def pooled(questions, ks=KS, cand=50):
    docs, owner = [], []
    for q in questions:
        for d in q["docs"]:
            docs.append({"doc_id": f'{d["doc_id"]}', "title": d["title"],
                         "text": d["text"], "sentences": d["sentences"]})
            owner.append(q["qid"])
    idx = QuestionIndex({"qid": "pool", "docs": docs})

    # natural duplication statistics
    pair_count = {}
    for d in docs:
        pair_count[(d["title"], d["text"])] = pair_count.get((d["title"], d["text"]), 0) + 1
    title_count = {}
    for d in docs:
        title_count[d["title"]] = title_count.get(d["title"], 0) + 1
    dup = sum(1 for v in pair_count.values() if v > 1)
    stats = {
        "n_docs": len(docs),
        "n_unique": len(pair_count),
        "n_texts_duplicated": dup,
        "max_copies": max(pair_count.values()),
        "n_titles_repeated": sum(1 for v in title_count.values() if v > 1),
    }

    curve = {k: {"pointwise": [], "coverage": [], "same_title": []} for k in ks}
    cand_recall = []
    for q in questions:
        scores = idx.score_docs(q["question"])
        cand_idx = pointwise(scores, cand)
        sub_scores = [scores[i] for i in cand_idx]
        gold = hop_gold(q)
        key2hop = {v: h for h, v in gold.items()}
        covers = []
        for i in cand_idx:
            d = docs[i]
            h = key2hop.get((d["title"], d["text"]))
            covers.append({h} if h is not None else set())
        n_hops = len(gold)
        cand_recall.append(len(set().union(*covers)) / n_hops)
        for k in ks:
            pw = pointwise(sub_scores, k)
            cv = greedy(sub_scores, covers, k)
            curve[k]["pointwise"].append(len(set().union(*[covers[i] for i in pw])) / n_hops)
            curve[k]["coverage"].append(len(set().union(*[covers[i] for i in cv])) / n_hops)
            titles = [docs[cand_idx[i]]["title"] for i in pw]
            curve[k]["same_title"].append(len(titles) - len(set(titles)) > 0)
    stats["cand_recall"] = {"mean": ev.mean(cand_recall), "ci": ev.bootstrap_ci(cand_recall)}
    return curve, stats


def summarize(curve, ks):
    return {k: {m: {"mean": ev.mean(v), "ci": ev.bootstrap_ci(v)} for m, v in d.items()}
            for k, d in curve.items()}


def report(pq, pool_sum, st, n_q, ks):
    lines = ["# E1: coverage objective on natural corpora (MuSiQue)", ""]
    lines.append(f"- questions: {n_q}　|　per-question pool: 19–20 paragraphs　|　pooled: {st['n_docs']} paragraphs")
    lines.append(f"- natural exact duplicates: {st['n_texts_duplicated']} texts appear more than once "
                 f"(max {st['max_copies']} copies); repeated titles: {st['n_titles_repeated']}")
    lines.append("")
    lines.append("## per-question pool: hop coverage@k (oracle labels)")
    lines.append("")
    lines.append("| k | pointwise | greedy coverage | pointwise redundancy |")
    lines.append("|---|---|---|---|")
    for k in ks:
        d = pq[k]
        lines.append(f"| {k} | {d['pointwise']['mean']:.3f} | {d['coverage']['mean']:.3f} | "
                     f"{d['redundancy']['mean']:.3f} |")
    lines.append("")
    lines.append("## pooled corpus (candidates = BM25 top-50)")
    lines.append("")
    lines.append(f"- candidate recall (share of hops whose gold is inside top-50): "
                 f"{st['cand_recall']['mean']:.3f} — this is the ceiling of any selector on this candidate set")
    lines.append("")
    lines.append("| k | pointwise | greedy coverage | top-k has same-title dup |")
    lines.append("|---|---|---|---|")
    for k in ks:
        d = pool_sum[k]
        lines.append(f"| {k} | {d['pointwise']['mean']:.3f} | {d['coverage']['mean']:.3f} | "
                     f"{d['same_title']['mean']:.3f} |")
    lines.append("")
    lines.append("> pointwise redundancy = share of selected docs that cover an already-covered hop.")
    path = os.path.join(RES, "e1_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[e1] wrote {path}")


def run(n=200):
    questions = load()[:n]
    pq = summarize(per_question(questions), KS)
    pc, st = pooled(questions, KS)
    pc = summarize(pc, KS)
    with open(os.path.join(RES, "e1_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"per_question": pq, "pooled": pc, "dup_stats": st}, f, ensure_ascii=False, indent=2)
    report(pq, pc, st, len(questions), KS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    run(ap.parse_args().n)
