"""Lookahead binding: choose a binding by how well the NEXT hop retrieves."""
import argparse, json, os, re, math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import fmt, topk as bm25_topk
from .llm import LLM
from .retriever import QuestionIndex
from .musique import load
from .data import load_questions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

READ = """Answer the sub-question using only the passages. Output only the answer span.

Original question: {q0}
Sub-question: {q}
Passages:
{ps}
Answer:"""
_cache = {"emb": {}}


def retrieve(ce, index, text, k=5):
    key = (id(index), len(index.docs))
    if key not in _cache["emb"]:
        _cache["emb"][key] = ce.embed([d["text"] for d in index.docs])
    de = _cache["emb"][key]
    cbm, _ = bm25_topk(index, text, 20)
    try:
        qe = ce.embed([text])[0]
        order = list(np.argsort(-(de @ qe)))[:20]
        cdn = [index.docs[i]["text"] for i in order]
    except Exception:
        cdn = []
    cand = list(dict.fromkeys(cbm + cdn))
    try:
        sc = ce.rerank(text, cand)
        o = sorted(range(len(cand)), key=lambda i: -sc[i])[:k]
        return [cand[i] for i in o], [sc[i] for i in o]
    except Exception:
        return cand[:k], [0.0] * min(k, len(cand))


def sig(x):
    return 1 / (1 + math.exp(-x))


def render(text, binds):
    for i, a in enumerate(binds, 1):
        if a:
            text = text.replace(f"#{i}", a)
    return re.sub(r"#\d+", " ", text).replace(">>", " ").strip()


def load_data(path):
    p = os.path.join(ROOT, "data", "raw", path)
    return load_questions(p) if path.startswith("hotpot") else load(p)


def main(n=200, workers=10, data="musique200.jsonl", policy="lookahead"):
    qs = load_data(data)[:n]
    docs = [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"],
             "sentences": d["sentences"]} for q in qs for d in q["docs"]]
    index = QuestionIndex({"qid": "pool", "docs": docs})
    ce = CERemote(); llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    tag = data.replace(".jsonl", "")

    plans = {}
    pp = os.path.join(RES, f"e14_plans_{tag}.jsonl")
    if os.path.exists(pp):
        for line in open(pp, encoding="utf-8"):
            r = json.loads(line); plans[r["qid"]] = r["hops"]

    path = os.path.join(RES, f"e20_{tag}_{policy}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[e20] {tag} {policy}: {len(todo)}")

    def work(q):
        hops = plans.get(q["qid"]) or [{"query": q["question"], "type": "other"}]
        binds, ev = [], []
        for h, hop in enumerate(hops[:4]):
            text = render(hop["query"], binds)
            ps, sc = retrieve(ce, index, text)
            try:
                concat = llm.chat([{"role": "user", "content": READ.format(q0=q["question"], q=text, ps=fmt(ps))}],
                                  max_tokens=32).strip().split("\n")[0]
            except Exception:
                concat = ""
            ans = concat
            if policy == "lookahead" and h + 1 < len(hops):
                cands = []
                try:
                    cands.append(concat)
                    for p in ps[:3]:
                        cands.append(llm.chat([{"role": "user", "content": READ.format(
                            q0=q["question"], q=text, ps=fmt([p]))}], max_tokens=32).strip().split("\n")[0])
                except Exception:
                    pass
                cands = [a for a in dict.fromkeys(cands) if a and "not contain" not in a.lower() and "cannot" not in a.lower()][:3]
                best, best_score = None, -1e9
                for a in cands:
                    nt = render(hops[h + 1]["query"], binds + [a])
                    _, sc2 = retrieve(ce, index, nt)
                    margin = (sig(sc2[0]) - sig(sc2[1])) if len(sc2) > 1 else sig(sc2[0])
                    if margin > best_score:
                        best, best_score = a, margin
                if best:
                    ans = best
            binds.append(ans); ev.extend(ps)
        seen, uniq = set(), []
        for p in ev:
            if p not in seen:
                seen.add(p); uniq.append(p)
        if binds and binds[-1] and "not contain" not in binds[-1].lower() and "cannot" not in binds[-1].lower():
            pred = binds[-1]
        else:
            try:
                pred = llm.chat([{"role": "user", "content": READ.format(q0=q["question"], q=q["question"], ps=fmt(uniq[:8]))}],
                                max_tokens=32).strip().split("\n")[0]
            except Exception:
                pred = binds[-1] if binds else ""
        rec = {"qid": q["qid"], "type": q.get("type", ""), "pred": pred, "gold": q["answer"],
               "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"]), "n_hops": len(hops)}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 25 == 0 or i == len(todo):
                print(f"[e20] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="musique200.jsonl")
    ap.add_argument("--policy", default="lookahead", choices=["base", "lookahead"])
    a = ap.parse_args()
    main(a.n, 10, a.data, a.policy)
