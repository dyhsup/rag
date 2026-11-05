"""Complexity router: single-hop -> vanilla path, multi-hop -> RBG execution.

The planner emits a typed decision `multi_hop`, so routing is part of the
program compilation rather than a string heuristic.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt, topk
from .e13_exec import solve
from .llm import LLM
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

PLAN = """Decide whether the question requires chaining facts across entities (multi-hop) or can be answered from a single passage (single-hop).
Then, if multi-hop, split it into 2-4 hop queries. Use "#1" to refer to the answer of hop 1.
If single-hop, return a single hop equal to the original question.
Output JSON only: {{"multi_hop": true/false, "hops": [{{"query": "...", "type": "..."}}]}}

Question: {q}
Output:"""


def plan(llm, q):
    try:
        obj = llm.json_call(PLAN.format(q=q["question"]), max_tokens=320)
        hops = [{"query": h["query"], "type": h.get("type", "other")}
                for h in obj.get("hops", []) if h.get("query")][:4]
        mh = bool(obj.get("multi_hop"))
        if not hops:
            hops = [{"query": q["question"], "type": "other"}]
        if not mh:
            hops = [{"query": q["question"], "type": "other"}]
        return {"multi_hop": mh, "hops": hops}
    except Exception:
        return {"multi_hop": False, "hops": [{"query": q["question"], "type": "other"}]}


def refine(p, q):
    """multi_hop claims must be backed by a real dependency edge."""
    hops, mh = p["hops"], p["multi_hop"]
    if mh and len(hops) > 1 and not any("#" in h["query"] for h in hops):
        qn = " " + q["question"].lower() + " "
        if not any(w in qn for w in [" both ", " same ", " or ", " compared ", " more ", " less ", " first "]):
            return {"multi_hop": False, "hops": [{"query": q["question"], "type": "other"}]}
    return {"multi_hop": mh, "hops": hops}


def load_data(path):
    from .data import load_questions
    from .musique import load
    if "hotpot" in path or "squad" in path:
        return load_questions(os.path.join(ROOT, "data", "raw", path))
    return load(os.path.join(ROOT, "data", "raw", path))


def main(n=200, workers=10, data="squad_200.jsonl"):
    qs = load_data(data)[:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    ce = CERemote()

    tag = data.replace(".jsonl", "")
    ppath = os.path.join(RES, f"e14_plans_{tag}.jsonl")
    plans = {}
    if os.path.exists(ppath):
        for line in open(ppath, encoding="utf-8"):
            r = json.loads(line)
            plans[r["qid"]] = r
    todo = [q for q in qs if q["qid"] not in plans]
    if todo:
        def mk(q):
            return {"qid": q["qid"], **plan(llm, q)}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(mk, q) for q in todo]):
                r = fut.result()
                plans[r["qid"]] = r
                with open(ppath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nh = sum(1 for r in plans.values() if r["multi_hop"])
    print(f"[e14] plans={len(plans)} multi_hop={nh}")

    path = os.path.join(RES, f"e14_router_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    run_qs = [q for q in qs if q["qid"] not in done]
    print(f"[e14] {len(run_qs)} to run")

    def work(q):
        p = refine(plans[q["qid"]], q)
        if p["multi_hop"]:
            pred, tr = solve(q, p["hops"], index, llm, ce, "select")
        else:
            ps, _ = topk(index, q["question"], 5)
            try:
                pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(ps))}],
                                max_tokens=32).strip().split("\n")[0]
            except Exception:
                pred = ""
            tr = []
        rec = {"qid": q["qid"], "pred": pred, "gold": q["answer"],
               "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"]),
               "multi_hop": p["multi_hop"], "n_hops": len(p["hops"])}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in run_qs]), 1):
            fut.result()
            if i % 25 == 0 or i == len(run_qs):
                print(f"[e14] {i}/{len(run_qs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="squad_200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
