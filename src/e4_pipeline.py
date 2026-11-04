"""E4: end-to-end pipeline vs strong baselines on MuSiQue.

Methods share corpus (pooled paragraphs), retriever (BM25) and reader (qwen):
  vanilla  : top-5 by the question, read once
  ircot    : CoT sentence -> retrieve, 3 rounds, read the union
  selfask  : free-form follow-up questions, each read, then final read
  ours     : plan -> bind predicted answers -> retrieve -> CE rerank -> read
             (--plan gold uses MuSiQue decomposition, auto uses the LLM)
"""
import argparse
import json
import os
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import eval as ev
from .ce_client import CERemote
from .e3_binding import em, f1, render
from .llm import LLM
from .data import load_questions
from .musique import load as load_musique
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_lock = threading.Lock()

READ = """Answer the question using only the passages. Output only the answer span.

Question: {q}
Passages:
{ps}
Answer:"""

IRCOT_NEXT = """Given the question and the passages retrieved so far, write ONE next reasoning sentence that moves toward the answer. Output only that sentence.

Question: {q}
Passages:
{ps}
Next reasoning sentence:"""

SELFASK = """Answer the question by asking and answering follow-up questions. Use exactly this format.

Question: Who is the spouse of the Green performer?
Are follow up questions needed here: Yes
Follow up: Who is the Green performer?
Intermediate answer: Steve Hillage
Follow up: Who is the spouse of Steve Hillage?
Intermediate answer: Miquette Giraudy
So the final answer is: Miquette Giraudy

Question: {q}
Are follow up questions needed here:"""

PLAN = """Split the question into 2-4 hop queries. Each hop must be answerable by one Wikipedia sentence.
Use "#1" to refer to the answer of hop 1, "#2" for hop 2, and so on.
Output JSON only: {{"hops": [{{"query": "..."}}]}}

Question: {q}
Output:"""


def fmt(passages):
    return "\n".join(f"[{i+1}] {p[:900]}" for i, p in enumerate(passages))


def read(llm, q, passages):
    try:
        out = llm.chat([{"role": "user", "content": READ.format(q=q, ps=fmt(passages))}],
                       max_tokens=32).strip()
        return out.split("\n")[0]
    except Exception:
        return ""


def topk(index, query, k):
    s = index.score_docs(query)
    order = sorted(range(len(s)), key=lambda i: -s[i])[:k]
    return [index.docs[i]["text"] for i in order], [float(s[i]) for i in order]


def run_vanilla(q, index, llm, ce):
    ps, _ = topk(index, q["question"], 5)
    return read(llm, q["question"], ps), {"retrieved": 5}


def run_ircot(q, index, llm, ce):
    ps, _ = topk(index, q["question"], 5)
    acc = list(ps)
    for _ in range(2):
        try:
            sent = llm.chat([{"role": "user", "content": IRCOT_NEXT.format(
                q=q["question"], ps=fmt(acc[-5:]))}], max_tokens=48).strip().split("\n")[0]
        except Exception:
            break
        if not sent:
            break
        ps, _ = topk(index, sent, 5)
        acc.extend(ps)
    seen, uniq = set(), []
    for p in acc:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return read(llm, q["question"], uniq[:8]), {"retrieved": len(uniq)}


def run_selfask(q, index, llm, ce):
    try:
        text = llm.chat([{"role": "user", "content": SELFASK.format(q=q["question"])}],
                        max_tokens=256).strip()
    except Exception:
        return read(llm, q["question"], topk(index, q["question"], 5)[0]), {"steps": 0}
    ctx, steps = [], 0
    for m in re.finditer(r"Follow up:\s*(.+?)\s*\nIntermediate answer:\s*(.+?)\s*\n", text + "\n"):
        sub, ans = m.group(1).strip(), m.group(2).strip()
        ps, _ = topk(index, sub, 3)
        ctx.append(f"Q: {sub}\nA: {ans}\nEvidence: " + " ".join(p[:300] for p in ps))
        steps += 1
    final = text.split("So the final answer is:")[-1].strip().split("\n")[0]
    if not final:
        final = read(llm, q["question"], topk(index, q["question"], 5)[0])
    return final, {"steps": steps}


def auto_plan(llm, q):
    try:
        obj = llm.json_call(PLAN.format(q=q["question"]), max_tokens=256)
        hops = [h["query"] for h in obj.get("hops", []) if h.get("query")]
        return hops[:4] or [q["question"]]
    except Exception:
        return [q["question"]]


def run_ours(q, index, llm, ce, plan_mode, fallback_tau=None, use_ce=True, use_bind=True):
    if plan_mode == "gold":
        hops = [h["question"] for h in q["hops"]]
        gold = [h["answer"] for h in q["hops"]]
    else:
        hops = auto_plan(llm, q)
        gold = None
    binds, evidence, trace = [], [], []
    for t in hops:
        if use_bind and binds:
            text = render(t, binds)
        else:
            text = render(t, [])
        cand, bm = topk(index, text, 50)
        if use_ce:
            try:
                scores = ce.rerank(text, cand)
            except Exception:
                scores = bm
        else:
            scores = bm
        order = sorted(range(len(cand)), key=lambda i: -scores[i])[:3]
        ps = [cand[i] for i in order]
        low_conf = False
        if fallback_tau is not None and scores:
            conf = 1 / (1 + math.exp(-scores[order[0]]))
            if conf < fallback_tau:
                low_conf = True
                extra, _ = topk(index, q["question"], 3)
                ps = ps + [x for x in extra if x not in ps][:2]
        ans = read(llm, text, ps)
        binds.append(ans)
        evidence.extend(ps)
        trace.append({"query": text, "answer": ans, "ce": [scores[i] for i in order], "fallback": low_conf})
    seen, uniq = set(), []
    for p in evidence:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    final = read(llm, q["question"], uniq[:8])
    return final, {"hops": hops, "trace": trace}


def one(q, method, index, llm, ce, plan_mode):
    if method == "vanilla":
        pred, info = run_vanilla(q, index, llm, ce)
    elif method == "ircot":
        pred, info = run_ircot(q, index, llm, ce)
    elif method == "selfask":
        pred, info = run_selfask(q, index, llm, ce)
    elif method == "ours_none":
        pred, info = run_ours(q, index, llm, ce, plan_mode, use_ce=False, use_bind=False)
    elif method == "ours_nobind":
        pred, info = run_ours(q, index, llm, ce, plan_mode, use_bind=False)
    elif method == "ours_bm25":
        pred, info = run_ours(q, index, llm, ce, plan_mode, use_ce=False)
    elif method == "ours_fb":
        pred, info = run_ours(q, index, llm, ce, plan_mode, fallback_tau=0.9)
    else:
        pred, info = run_ours(q, index, llm, ce, plan_mode)
    return {"qid": q["qid"], "method": method, "pred": pred, "gold": q["answer"],
            "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"]), "info": info}


def fname(method, plan_mode, dataset):
    return (f"e4_{dataset}_{method}_{plan_mode}.jsonl" if method.startswith("ours")
            else f"e4_{dataset}_{method}.jsonl")


def run(n, workers, methods, plan_mode, dataset="musique"):
    qs = (load_musique() if dataset == "musique" else load_questions(
        os.path.join(ROOT, "data", "raw", "hotpot200.jsonl")))[:n]
    docs = []
    for q in qs:
        for d in q["docs"]:
            docs.append({"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
                         "sentences": d["sentences"]})
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()

    for method in methods:
        path = os.path.join(RES, fname(method, plan_mode, dataset))
        done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                done = {json.loads(l)["qid"] for l in f if l.strip()}
        todo = [q for q in qs if q["qid"] not in done]
        print(f"[{method}] {len(todo)} to run ({len(done)} cached)", flush=True)

        def work(q):
            rec = one(q, method, index, llm, ce, plan_mode)
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            return rec

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
                fut.result()
                if i % 20 == 0 or i == len(todo):
                    print(f"[{method}] {i}/{len(todo)} calls={llm.calls}", flush=True)
    summarize(methods, plan_mode, dataset)


def summarize(methods, plan_mode, dataset):
    stats = {}
    for m in methods:
        path = os.path.join(RES, fname(m, plan_mode, dataset))
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        stats[m] = {"em": ev.mean([r["em"] for r in recs]),
                    "f1": ev.mean([r["f1"] for r in recs]),
                    "em_ci": ev.bootstrap_ci([r["em"] for r in recs]),
                    "f1_ci": ev.bootstrap_ci([r["f1"] for r in recs]),
                    "n": len(recs)}
    lines = [f"# E4 end-to-end: pipeline vs baselines ({dataset})", "",
             "| method | EM [95% CI] | F1 [95% CI] | n |", "|---|---|---|---|"]
    for m, d in stats.items():
        lines.append(f"| {m} | {d['em']:.3f} [{d['em_ci'][1]:.3f},{d['em_ci'][2]:.3f}] | "
                     f"{d['f1']:.3f} [{d['f1_ci'][1]:.3f},{d['f1_ci'][2]:.3f}] | {d['n']} |")
    # paired vs vanilla
    if "vanilla" in stats:
        base = {json.loads(l)["qid"]: json.loads(l)["em"]
                for l in open(os.path.join(RES, fname("vanilla", plan_mode, dataset)), encoding="utf-8") if l.strip()}
        lines += ["", "| method | ΔEM vs vanilla [95% CI] |", "|---|---|"]
        for m in stats:
            if m == "vanilla":
                continue
            cur = {json.loads(l)["qid"]: json.loads(l)["em"]
                   for l in open(os.path.join(RES, fname(m, plan_mode, dataset)), encoding="utf-8") if l.strip()}
            qids = [k for k in base if k in cur]
            d = ev.paired_bootstrap([cur[k] for k in qids], [base[k] for k in qids])
            lines.append(f"| {m} | {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] |")
    lines.append("")
    with open(os.path.join(RES, f"e4_report_{dataset}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, f"e4_summary_{dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", default="vanilla,ircot,selfask,ours")
    ap.add_argument("--plan", default="gold", choices=["gold", "auto"])
    ap.add_argument("--dataset", default="musique", choices=["musique", "hotpot"])
    a = ap.parse_args()
    ms = [m.strip() for m in a.methods.split(",") if m.strip()]
    run(a.n, a.workers, ms, a.plan, a.dataset)
