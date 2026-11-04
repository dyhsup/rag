"""E5: measure and close the query-evidence semantic gap. Retrieval metrics only.

Setup: chunk-level corpus (2-sentence windows), oracle binding on the query side,
so multi-hop/binding is fixed and not a variable. Gold chunk = a chunk from the
hop's supporting paragraph that contains the hop answer.

Stage measure : rank of the gold chunk under BM25 and dense; gap share.
Stage close   : paraphrase projection (3 variants, max-score merge) and how many
                gap cases it pulls back into top-3.
"""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import eval as ev
from .ce_client import CERemote
from .e2b_ce import render_hop
from .llm import LLM
from .musique import load
from .retriever import QuestionIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
IDX = os.path.join(ROOT, "data", "index")
_lock = threading.Lock()

CLAIM = """Rewrite the search query as ONE declarative statement that the evidence would contain, using "?" for the answer.
Do not add facts. Output JSON only: {{"claim": "..."}}

Query: {q}
Output:"""

PARA = """Rewrite the search query in 3 different ways that mean exactly the same thing.
Change the wording of the relation and, when useful, replace an entity with a descriptive phrase.
Do not add or remove facts. Output JSON only: {{"queries": ["...", "...", "..."]}}

Query: {q}
Output:"""


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def chunks_of(para_text, width=40, stride=20):
    """Fixed token windows; avoids sentence-splitter artifacts (e.g. 'c. 1450')."""
    toks = para_text.split()
    if len(toks) <= width:
        return [" ".join(toks)] if len(toks) >= 8 else []
    out = []
    for i in range(0, len(toks) - width + 1, stride):
        out.append(" ".join(toks[i:i + width]))
    if (len(toks) - width) % stride:
        out.append(" ".join(toks[-width:]))
    return out


def build(questions):
    corpus, meta = [], []
    gold = {}
    queries = []
    for q in questions:
        for d in q["docs"]:
            cs = chunks_of(d["text"])
            for ci, c in enumerate(cs):
                corpus.append(c)
                meta.append({"qid": q["qid"], "para_idx": d["idx"], "ci": ci})
        for h in q["hops"]:
            ans = norm(h["answer"])
            ids = [i for i, m in enumerate(meta)
                   if m["qid"] == q["qid"] and m["para_idx"] == h["para_idx"]
                   and ans and re.search(r"\b" + re.escape(ans) + r"\b", norm(corpus[i]))]
            gold[(q["qid"], h["hop"])] = ids
            queries.append({"qid": q["qid"], "hop": h["hop"],
                            "text": render_hop(q, h["hop"])})
    return corpus, meta, gold, queries


def ranks(scores, ids):
    if not ids:
        return None
    best = min(int((scores > scores[i]).sum()) + 1 for i in ids)
    return best


def measure(questions, ce, chunk_emb=None):
    corpus, meta, gold, queries = build(questions)
    print(f"[e5] chunks={len(corpus)} queries={len(queries)}")
    index = QuestionIndex({"qid": "pool", "docs": [
        {"doc_id": str(i), "title": "", "text": c, "sentences": [c]} for i, c in enumerate(corpus)]})

    if chunk_emb is None:
        path = os.path.join(IDX, f"e5_chunks_{len(corpus)}.npy")
        if os.path.exists(path):
            chunk_emb = np.load(path)
        else:
            print("[e5] embedding chunks on the remote service")
            chunk_emb = ce.embed(corpus)
            np.save(path, chunk_emb)
    q_texts = [x["text"] for x in queries]
    q_emb = ce.embed(q_texts)

    rows = []
    for i, x in enumerate(queries):
        ids = gold[(x["qid"], x["hop"])]
        bm = np.array(index.score_docs(x["text"]))
        dn = chunk_emb @ q_emb[i]
        rows.append({**x, "n_gold": len(ids),
                     "rank_bm25": ranks(bm, ids), "rank_dense": ranks(dn, ids),
                     "bm25_scores": bm, "dense_scores": dn})
    return rows, corpus, meta, chunk_emb


def summarize(rows):
    out = {}
    for name in ["rank_bm25", "rank_dense"]:
        vals = [r[name] for r in rows if r[name] is not None]
        out[name] = {
            "n": len(vals),
            "r@1": sum(1 for r in vals if r <= 1) / len(vals),
            "r@3": sum(1 for r in vals if r <= 3) / len(vals),
            "r@10": sum(1 for r in vals if r <= 10) / len(vals),
            "r@50": sum(1 for r in vals if r <= 50) / len(vals),
            "median": float(np.median(vals)),
        }
        gaps = [r for r in vals if 3 < r <= 50]
        out[name]["gap_share"] = len(gaps) / len(vals)
    return out


def gen_claims(queries, llm, workers=8):
    path = os.path.join(RES, "e5_claim.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[(r["qid"], r["hop"])] = r.get("claim", "")
    todo = [x for x in queries if (x["qid"], x["hop"]) not in done]
    print(f"[e5] claims {len(todo)}")
    def work(x):
        try:
            obj = llm.json_call(CLAIM.format(q=x["text"]), max_tokens=128)
            c = (obj.get("claim") or "").strip()
        except Exception:
            c = ""
        return {"qid": x["qid"], "hop": x["hop"], "claim": c}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, x) for x in todo]), 1):
            rec = fut.result()
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"[e5] claims {i}/{len(todo)}", flush=True)
    return done


def paraphrase(queries, llm, workers=8):
    path = os.path.join(RES, "e5_paraphrase.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[(r["qid"], r["hop"])] = r
    todo = [x for x in queries if (x["qid"], x["hop"]) not in done]
    print(f"[e5] paraphrasing {len(todo)} queries")

    def work(x):
        try:
            obj = llm.json_call(PARA.format(q=x["text"]), max_tokens=256)
            qs = [s for s in obj.get("queries", []) if isinstance(s, str) and s.strip()][:3]
        except Exception:
            qs = []
        return {"qid": x["qid"], "hop": x["hop"], "text": x["text"], "paras": qs}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, x) for x in todo]), 1):
            rec = fut.result()
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"[e5] {i}/{len(todo)}", flush=True)
    return done


def close(rows, corpus, chunk_emb, ce, claims=None, para_text=None):
    path = os.path.join(RES, "e5_paraphrase.jsonl")
    paras = {(r["qid"], r["hop"]): r["paras"] for r in
             (json.loads(l) for l in open(path, encoding="utf-8")) if r["paras"]}
    texts, keys = [], []
    for r in rows:
        k = (r["qid"], r["hop"])
        for p in paras.get(k, []):
            texts.append(p)
            keys.append(k)
    print(f"[e5] embedding {len(texts)} paraphrases")
    emb = ce.embed(texts) if texts else np.zeros((0, chunk_emb.shape[1]), dtype=np.float32)
    by_key = {}
    for e, k in zip(emb, keys):
        by_key.setdefault(k, []).append(e)

    base_r3 = sum(1 for r in rows if r["rank_dense"] is not None and r["rank_dense"] <= 3)
    gap_rows = [r for r in rows if r["rank_dense"] is not None and 3 < r["rank_dense"] <= 50]
    closed_para = closed_ce = closed_claim = closed_ctx = 0
    details = []
    for r in gap_rows:
        k = (r["qid"], r["hop"])
        ids = _gold_ids(r, rows)

        s = r["dense_scores"].copy()
        for e in by_key.get(k, []):
            s = np.maximum(s, chunk_emb @ e)
        rk_para = ranks(s, ids)

        top = np.argsort(-r["dense_scores"])[:50]
        try:
            ce_scores = ce.rerank(r["text"], [corpus[i] for i in top])
        except Exception:
            ce_scores = [float(r["dense_scores"][i]) for i in top]
        merged = r["dense_scores"].copy()
        for j, i in enumerate(top):
            merged[i] = ce_scores[j]
        rk_ce = ranks(merged, ids)

        cl = (claims or {}).get(k) or r["text"]
        try:
            cl_scores = ce.rerank(cl, [corpus[i] for i in top])
        except Exception:
            cl_scores = ce_scores
        merged_cl = r["dense_scores"].copy()
        for j, i in enumerate(top):
            merged_cl[i] = cl_scores[j]
        rk_cl = ranks(merged_cl, ids)

        # evidence-side context restoration: chunk score + its parent paragraph score
        try:
            ptxt = (para_text or {}).get(k, [])
            par_scores = ce.rerank(r["text"], [ptxt[meta_i["para_idx"]] if ptxt else corpus[i] for meta_i, i in zip([META[i] for i in top], top)])
        except Exception:
            par_scores = ce_scores
        merged_ctx = r["dense_scores"].copy()
        for j, i in enumerate(top):
            merged_ctx[i] = 0.5 * ce_scores[j] + 0.5 * par_scores[j]
        rk_ctx = ranks(merged_ctx, ids)
        closed_ctx += int(rk_ctx is not None and rk_ctx <= 3)

        closed_para += int(rk_para is not None and rk_para <= 3)
        closed_ce += int(rk_ce is not None and rk_ce <= 3)
        closed_claim += int(rk_cl is not None and rk_cl <= 3)
        details.append({"qid": r["qid"], "hop": r["hop"], "before": r["rank_dense"],
                        "after_paraphrase": rk_para, "after_ce": rk_ce,
                        "after_claim": rk_cl, "after_ctx": rk_ctx})
    n = len(gap_rows)
    return {"gap_cases": n,
            "closed_paraphrase": closed_para, "closure_rate_paraphrase": closed_para / n if n else float("nan"),
            "closed_ce": closed_ce, "closure_rate_ce": closed_ce / n if n else float("nan"),
            "closed_claim": closed_claim, "closure_rate_claim": closed_claim / n if n else float("nan"),
            "closed_ctx": closed_ctx, "closure_rate_ctx": closed_ctx / n if n else float("nan"),
            "base_r@3": base_r3 / len(rows), "details": details}


_GOLD = {}
META = []


def _gold_ids(r, rows):
    return _GOLD.get((r["qid"], r["hop"]), [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stage", default="all", choices=["measure", "close", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    qs = load()[:a.n]
    ce = CERemote()
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    rows, corpus, meta, chunk_emb = measure(qs, ce)
    global _GOLD, META
    META = meta
    _, _, _GOLD, _ = build(qs)
    s = summarize(rows)
    print(json.dumps(s, ensure_ascii=False, indent=2))

    result = {"summary": s, "n_chunks": len(corpus), "n_queries": len(rows)}
    if a.stage in ("close", "all"):
        paraphrase([{"qid": r["qid"], "hop": r["hop"], "text": r["text"]} for r in rows],
                   llm, a.workers)
        gen_claims([{"qid": r["qid"], "hop": r["hop"], "text": r["text"]} for r in rows],
                   llm, a.workers)
        claims = {}
        if os.path.exists(os.path.join(RES, "e5_claim.jsonl")):
            for line in open(os.path.join(RES, "e5_claim.jsonl"), encoding="utf-8"):
                x = json.loads(line)
                if x.get("claim"):
                    claims[(x["qid"], x["hop"])] = x["claim"]
        para_text = {}
        for q in qs:
            para_text[(q["qid"], -1)] = [d["text"] for d in q["docs"]]
        c = close(rows, corpus, chunk_emb, ce, claims, para_text)
        result["closure"] = {k: v for k, v in c.items() if k != "details"}
        result["closure_details"] = c["details"][:50]
        print(json.dumps(result["closure"], ensure_ascii=False, indent=2))

    with open(os.path.join(RES, "e5_gap.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
