"""Pilot driver.

Usage (from /disk1/query-evidance):
    python3 -m src.run_pilot --stage all

Stages: data -> claims (1 LLM call/question) -> judge (1 call/passage, B-line
upper bound) -> report. Results are appended incrementally so a long run can be
inspected while it is still going.
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import data as data_mod
from . import eval as ev
from .claims import project
from .coverage import score_variants
from .llm import LLM
from .retriever import QuestionIndex
from .slotjudge import judge

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_lock = threading.Lock()


# --------------------------------------------------------------------------
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rec):
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


# --------------------------------------------------------------------------
def stage_data(limit):
    path, rows = data_mod.build(n_questions=limit)
    print(f"[data] {path}: {len(rows)} questions")


def stage_claims(questions, llm, workers):
    path = os.path.join(RES, "claims.jsonl")
    done = {r["qid"] for r in read_jsonl(path)}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[claims] {len(todo)} to run ({len(done)} cached)")

    def work(q):
        try:
            return {"qid": q["qid"], "question": q["question"], **project(llm, q["question"]), "error": None}
        except Exception as e:
            return {"qid": q["qid"], "question": q["question"], "claims": [], "answer_type": None, "error": repr(e)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, q) for q in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            append_jsonl(path, rec)
            if i % 5 == 0 or i == len(futs):
                print(f"[claims] {i}/{len(futs)}  calls={llm.calls} hits={llm.cache_hits}", flush=True)


def stage_judge(questions, llm, workers):
    path = os.path.join(RES, "judge.jsonl")
    claims_by_q = {r["qid"]: r for r in read_jsonl(os.path.join(RES, "claims.jsonl"))}
    done = {(r["qid"], r["doc_id"]) for r in read_jsonl(path)}
    tasks = []
    for q in questions:
        c = claims_by_q.get(q["qid"], {})
        if not c.get("claims"):
            continue
        for d in q["docs"]:
            if (q["qid"], d["doc_id"]) not in done:
                tasks.append((q, c["claims"], d))
    print(f"[judge] {len(tasks)} (question, passage) pairs to run ({len(done)} cached)")

    def work(task):
        q, claims, d = task
        try:
            r = judge(llm, q["question"], claims, d)
            r["error"] = None
        except Exception as e:
            r = {"covered": [], "covered_count": 0, "score": 0.0, "span_total": 0, "span_valid": 0, "error": repr(e)}
        r.update({"qid": q["qid"], "doc_id": d["doc_id"]})
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            append_jsonl(path, fut.result())
            if i % 20 == 0 or i == len(futs):
                print(f"[judge] {i}/{len(futs)}  calls={llm.calls} hits={llm.cache_hits}", flush=True)


# --------------------------------------------------------------------------
def stage_report(questions, llm):
    claims_by_q = {r["qid"]: r for r in read_jsonl(os.path.join(RES, "claims.jsonl"))}
    judge_by_q = {}
    for r in read_jsonl(os.path.join(RES, "judge.jsonl")):
        judge_by_q.setdefault(r["qid"], {})[r["doc_id"]] = r

    variants = [
        "bm25_whole",
        "claim_max",
        "claim_mean",
        "claim_min",
        "claim_cover",
        "claim_cover_tb",
        "hybrid_0.3",
        "hybrid_0.5",
    ]
    if judge_by_q:
        variants += ["judge_oracle", "judge_tb"]

    per_q = {v: {"mrr": [], "ndcg": [], "r1": [], "r2": [], "auc_gold": [], "auc_ans": []} for v in variants}
    rows = []

    for q in questions:
        c = claims_by_q.get(q["qid"], {})
        claims = c.get("claims", [])
        if not claims:
            continue
        idx = QuestionIndex(q)
        scores, _per_claim = score_variants(idx, q["question"], claims)

        if q["qid"] in judge_by_q:
            jm = judge_by_q[q["qid"]]
            jscore = [jm.get(d["doc_id"], {}).get("score", 0.0) for d in q["docs"]]
            scores["judge_oracle"] = jscore
            # The judge is often all-zero on a question; without a tie-break the
            # ranking is just the passage order. Break ties with BM25 instead.
            scores["judge_tb"] = [
                jscore[i] + 1e-3 * scores["bm25_whole"][i] for i in range(len(jscore))
            ]

        gold = {i for i, d in enumerate(q["docs"]) if d["is_gold"]}
        ans = [1 if d["contains_answer"] else 0 for d in q["docs"]]
        gold_lab = [1 if d["is_gold"] else 0 for d in q["docs"]]

        row = {"qid": q["qid"], "question": q["question"], "n_claims": len(claims),
               "claims": claims, "gold_titles": q["gold_titles"], "methods": {}}
        for v in variants:
            if v not in scores:
                continue
            s = scores[v]
            per_q[v]["mrr"].append(ev.mrr_at_k(s, gold))
            per_q[v]["ndcg"].append(ev.ndcg_at_k(s, gold))
            per_q[v]["r1"].append(ev.recall_at_k(s, gold, 1))
            per_q[v]["r2"].append(ev.recall_at_k(s, gold, 2))
            per_q[v]["auc_gold"].append(ev.auc(s, gold_lab))
            per_q[v]["auc_ans"].append(ev.auc(s, ans))
            order = sorted(range(len(s)), key=lambda i: (-s[i], i))
            row["methods"][v] = {
                "mrr": ev.mrr_at_k(s, gold),
                "top1": q["docs"][order[0]]["title"],
                "top1_is_gold": order[0] in gold,
                "first_gold_rank": next((r for r, i in enumerate(order, 1) if i in gold), -1),
            }
        rows.append(row)

    # aggregate
    table = {}
    for v in variants:
        d = per_q[v]
        table[v] = {
            m: {
                "mean": ev.mean(d[m]),
                "ci": ev.bootstrap_ci(d[m]),
            }
            for m in ["mrr", "ndcg", "r1", "r2", "auc_gold", "auc_ans"]
        }

    # paired significance vs bm25_whole
    paired = {}
    for v in variants:
        if v == "bm25_whole":
            continue
        paired[v] = {
            "mrr": ev.paired_bootstrap(per_q[v]["mrr"], per_q["bm25_whole"]["mrr"]),
            "ndcg": ev.paired_bootstrap(per_q[v]["ndcg"], per_q["bm25_whole"]["ndcg"]),
            "auc_gold": ev.paired_bootstrap(per_q[v]["auc_gold"], per_q["bm25_whole"]["auc_gold"]),
        }

    n_claims = [r["n_claims"] for r in rows]
    judge_recs = read_jsonl(os.path.join(RES, "judge.jsonl"))
    span_total = sum(r.get("span_total", 0) for r in judge_recs)
    span_valid = sum(r.get("span_valid", 0) for r in judge_recs)
    n_err = sum(1 for r in judge_recs if r.get("error"))
    claims_err = sum(1 for r in read_jsonl(os.path.join(RES, "claims.jsonl")) if r.get("error"))

    summary = {
        "n_questions": len(rows),
        "n_claims_mean": ev.mean(n_claims),
        "claims_errors": claims_err,
        "judge_calls": len(judge_recs),
        "judge_errors": n_err,
        "span_valid_rate": span_valid / span_total if span_total else float("nan"),
        "llm_calls": 0,
        "llm_cache_hits": _cached_response_count(),
        "table": table,
        "paired_vs_bm25": paired,
    }
    with open(os.path.join(RES, "pilot_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": rows}, f, ensure_ascii=False, indent=2)

    write_markdown(summary, per_q)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    return summary


def _cached_response_count():
    import sqlite3

    path = os.path.join(ROOT, "cache", "llm.sqlite")
    if not os.path.exists(path):
        return 0
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
    finally:
        con.close()


def _fmt(cell):
    m, (_, lo, hi) = cell["mean"], cell["ci"]
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]" if m == m else "-"


def write_markdown(summary, per_q):
    lines = ["# 20 题 pilot 结果", ""]
    lines.append(f"- 题目数：{summary['n_questions']}")
    lines.append(f"- 平均 claim 数：{summary['n_claims_mean']:.2f}")
    lines.append(f"- claim 投影失败：{summary['claims_errors']}")
    lines.append(f"- 槽位判定调用 / 失败：{summary['judge_calls']} / {summary['judge_errors']}")
    lines.append(f"- span 逐字校验通过率：{summary['span_valid_rate']:.3f}")
    lines.append(f"- 缓存响应总数（去重后的唯一调用）：{summary['llm_cache_hits']}")
    lines.append("")
    lines.append("## 指标（均值 [95% CI]，按题 bootstrap）")
    lines.append("")
    lines.append("| 方法 | MRR@10 | nDCG@10 | Recall@1 | Recall@2 | AUC(gold) | AUC(answer) |")
    lines.append("|---|---|---|---|---|---|---|")
    for v, t in summary["table"].items():
        lines.append(
            f"| {v} | {_fmt(t['mrr'])} | {_fmt(t['ndcg'])} | {_fmt(t['r1'])} | "
            f"{_fmt(t['r2'])} | {_fmt(t['auc_gold'])} | {_fmt(t['auc_ans'])} |"
        )
    lines.append("")
    lines.append("## 与 BM25 的配对差（均值 [95% CI]，不含 0 即视为有信号）")
    lines.append("")
    lines.append("| 方法 | ΔMRR@10 | ΔnDCG@10 | ΔAUC(gold) |")
    lines.append("|---|---|---|---|")
    for v, t in summary["paired_vs_bm25"].items():
        cells = []
        for k in ["mrr", "ndcg", "auc_gold"]:
            m, lo, hi = t[k]
            cells.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]" if m == m else "-")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")

    # cases where coverage fixed BM25's top-1
    path = os.path.join(RES, "pilot_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {path}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "claims", "judge", "report", "all"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.stage in ("data", "all"):
        stage_data(args.limit)

    questions = data_mod.load_questions()
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    if args.stage in ("claims", "all"):
        stage_claims(questions, llm, args.workers)
    if args.stage in ("judge", "all"):
        stage_judge(questions, llm, args.workers)
    if args.stage in ("report", "all"):
        stage_report(questions, llm)


if __name__ == "__main__":
    main()
