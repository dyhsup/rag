"""G1: query-program execution with a binding gate and rollback.

Modes:
  base      : bind every answer (equivalent to E4 ours_auto)
  gate      : bind only if the answer passes (verbatim, margin, type)
  rollback  : on a hop that fails the gate, roll back to the previous hop and
              try its next-best answer, then re-execute the failed hop

Primary answer per hop = reader over the concatenated top-3 passages (same as
E4). Alternatives = readers over top-1 and top-2 alone.
"""
import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt, topk
from .llm import LLM
from .data import load_questions
from .musique import load
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

READ_CTX = """Answer the sub-question using only the passages.
The original question is provided for context; the answer must answer the sub-question.
Output only the answer span.

Original question: {q0}
Sub-question: {q}
Passages:
{ps}
Answer:"""

PLAN = """Split the question into 2-4 hop queries. For each hop give the query and the answer type.
Use "#1" to refer to the answer of hop 1, "#2" for hop 2, and so on.
Answer type is one of: person, date, number, location, organization, title, boolean, other.
Output JSON only: {{"hops": [{{"query": "...", "type": "..."}}]}}

Question: {q}
Output:"""

MONTHS = ("january february march april may june july august september october november december")


def sig(x):
    return 1 / (1 + math.exp(-x))


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def type_ok(ans, atype):
    a = norm(ans)
    if not a:
        return False
    if atype == "date":
        return bool(re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", a)) or any(m in a for m in MONTHS.split())
    if atype == "number":
        return bool(re.search(r"\d", a))
    return True


def _jac(a, b):
    A, B = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(A & B) / max(1, len(A | B))


def simplify(hops, q):
    """Compile-time check: independent hops + non-comparison question -> one hop."""
    if len(hops) > 1 and not any("#" in h["query"] for h in hops):
        qn = " " + q["question"].lower() + " "
        if not any(w in qn for w in [" and ", " both ", " or ", " versus ", " compared "]):
            return [{"query": q["question"], "type": "other"}]
    return hops


def auto_plan(llm, q):
    try:
        obj = llm.json_call(PLAN.format(q=q["question"]), max_tokens=320)
        hops = [{"query": h["query"], "type": h.get("type", "other")}
                for h in obj.get("hops", []) if h.get("query")][:4]
        hops = hops or [{"query": q["question"], "type": "other"}]
        # compile-time simplification: no dependency edges + high overlap with the
        # original question means the "plan" is not a chain; collapse to one hop.
        return simplify(hops, q)
    except Exception:
        return [{"query": q["question"], "type": "other"}]


def render(text, binds):
    for i, a in enumerate(binds, 1):
        if a:
            text = text.replace(f"#{i}", a)
    return re.sub(r"#\d+", " ", text).replace(">>", " ").strip()


def deps_of(text):
    return [int(x) for x in re.findall(r"#(\d+)", text)]


def execute_once(q, hops, index, llm, ce, mode, alt=None):
    alt = alt or {}
    binds = [None] * len(hops)
    evidence, trace = [], []
    for h, hop in enumerate(hops):
        deps = deps_of(hop["query"])
        text = render(hop["query"], binds)
        if mode != "base" and any(d - 1 < h and binds[d - 1] is None for d in deps):
            text = (text + " " + q["question"]).strip()
        cand, _ = topk(index, text, 50)
        try:
            scores = ce.rerank(text, cand)
        except Exception:
            scores = [0.0] * len(cand)
        order = sorted(range(len(cand)), key=lambda i: -scores[i])[:3]
        ps = [cand[i] for i in order]
        sc = [scores[i] for i in order]

        cands = []
        try:
            prompt = (READ_CTX.format(q0=q["question"], q=text, ps=fmt(ps))
                      if mode in ("ctxread", "ctxlast", "select") else READ.format(q=text, ps=fmt(ps)))
            cands.append(llm.chat([{"role": "user", "content": prompt}],
                                  max_tokens=32).strip().split("\n")[0])
        except Exception:
            cands.append("")
        for j in (0, 1):
            if j < len(ps):
                try:
                    cands.append(llm.chat([{"role": "user", "content": READ.format(q=text, ps=fmt([ps[j]]))}],
                                          max_tokens=32).strip().split("\n")[0])
                except Exception:
                    cands.append("")

        if mode in ("base", "consensus", "ctxread", "ctxlast", "select"):
            if mode == "consensus" and len(cands) >= 3:
                cnt = {}
                for j, a in enumerate(cands):
                    k = norm(a)
                    if k:
                        cnt.setdefault(k, []).append(j)
                best = max(cnt.values(), key=len) if cnt else []
                if len(best) >= 2:
                    ans, used = cands[best[0]], best[0]
                else:
                    ans, used = cands[0], 0
            else:
                ans, used = cands[0], 0
            ok = True
        else:
            forced = alt.get(h)
            order_idx = [forced] if forced is not None else list(range(len(cands)))
            ans, ok, used = None, False, None
            for j in order_idx:
                if j is None or j >= len(cands):
                    continue
                a = cands[j]
                good = (bool(norm(a)) and any(norm(a) in norm(p) for p in ps)
                        and ((sig(sc[0]) - sig(sc[1])) if len(sc) > 1 else 1.0) >= 0.05
                        and type_ok(a, hop.get("type", "other")))
                if good:
                    ans, ok, used = a, True, j
                    break
            if ans is None:
                used = forced if (forced is not None and forced < len(cands)) else 0
                ans = cands[used]
        trace.append({"hop": h, "query": text, "answer": ans, "gate": ok, "idx": used,
                      "cands": cands, "scores": sc})
        binds[h] = ans if (ok or mode == "base") else None
        evidence.extend(ps)
    return binds, evidence, trace


def solve(q, hops, index, llm, ce, mode):
    binds, ev, tr = execute_once(q, hops, index, llm, ce, mode)
    if mode == "rollback":
        failed = [t["hop"] for t in tr if not t["gate"]]
        if failed:
            f = failed[0]
            alt = {f: 1}
            if f > 0:
                alt[f - 1] = 1
            b2, e2, t2 = execute_once(q, hops, index, llm, ce, mode, alt)
            if sum(t["gate"] for t in t2) > sum(t["gate"] for t in tr):
                binds, ev, tr = b2, e2, t2
    seen, uniq = set(), []
    for p in ev:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if mode == "ctxlast" and binds and binds[-1] and "cannot" not in binds[-1].lower() and "not contain" not in binds[-1].lower():
        final = binds[-1]
    elif mode == "select":
        last = binds[-1] if binds else ""
        try:
            synth = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(uniq[:8]))}],
                             max_tokens=32).strip().split("\n")[0]
        except Exception:
            synth = ""
        if not last:
            final = synth
        elif norm(last) == norm(synth) or not synth:
            final = last
        else:
            swap = sum(ord(c) for c in q["qid"]) % 2 == 1
            a, b = (synth, last) if swap else (last, synth)
            try:
                pick = llm.chat([{"role": "user", "content":
                    f"Question: {q['question']}\nAnswer A: {a}\nAnswer B: {b}\n"
                    "Which answer is correct? Reply with exactly one letter: A or B.\nAnswer:"}],
                    max_tokens=8).strip().upper()
            except Exception:
                pick = "A"
            chosen = b if "B" in pick[:3] else a
            final = chosen
    else:
        try:
            final = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(uniq[:8]))}],
                             max_tokens=32).strip().split("\n")[0]
        except Exception:
            final = binds[-1] if binds else ""
    return final, tr


def run(n=200, workers=8, modes=("base", "gate", "rollback"), data="musique200.jsonl"):
    path = os.path.join(ROOT, "data", "raw", data)
    qs = (load_questions(path) if data.startswith("hotpot") else load(path))[:n]
    docs = []
    for q in qs:
        for d in q["docs"]:
            docs.append({"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
                         "sentences": d["sentences"]})
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()

    plans = {}
    ppath = os.path.join(RES, f"e13_plans_{data.replace('.jsonl','')}.jsonl")
    if os.path.exists(ppath):
        for line in open(ppath, encoding="utf-8"):
            r = json.loads(line)
            plans[r["qid"]] = r["hops"]
    missing = [q for q in qs if q["qid"] not in plans]
    if missing:
        def mk(q):
            return {"qid": q["qid"], "hops": auto_plan(llm, q)}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(mk, q) for q in missing]):
                r = fut.result()
                plans[r["qid"]] = r["hops"]
                with open(ppath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[e13] plans={len(plans)}")

    stats = {}
    for mode in modes:
        path = os.path.join(RES, f"e13_{mode}_{data.replace('.jsonl','')}.jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                done.add(json.loads(line)["qid"])
        todo = [q for q in qs if q["qid"] not in done]
        print(f"[e13] {mode}: {len(todo)} to run ({len(done)} cached)")

        def work(q):
            hops = simplify(plans[q["qid"]], q)
            final, tr = solve(q, hops, index, llm, ce, mode)
            rec = {"qid": q["qid"], "hops": hops, "final": final, "gold": q["answer"],
                   "em": em(final, q["answer"]), "f1": f1(final, q["answer"]),
                   "trace": tr, "gate_pass": sum(t["gate"] for t in tr), "n_hops": len(tr)}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
                fut.result()
                if i % 20 == 0 or i == len(todo):
                    print(f"[e13] {mode} {i}/{len(todo)}", flush=True)

        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        by_hop = {}
        for r in recs:
            by_hop.setdefault(len(r["hops"]), []).append(r)
        stats[mode] = {
            "all": {"n": len(recs), "em": sum(r["em"] for r in recs) / len(recs)},
            "by_hops": {k: {"n": len(v), "em": sum(x["em"] for x in v) / len(v)} for k, v in by_hop.items()},
            "gate_rate": sum(r["gate_pass"] for r in recs) / max(1, sum(r["n_hops"] for r in recs)),
        }
    json.dump(stats, open(os.path.join(RES, f"e13_summary_{data.replace('.jsonl','')}.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--modes", default="base,gate,rollback")
    ap.add_argument("--data", default="musique200.jsonl")
    a = ap.parse_args()
    run(a.n, a.workers, tuple(a.modes.split(",")), a.data)
