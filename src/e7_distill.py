"""E7: distill the LLM gap judge into a small cross-encoder.

Stage collect: LLM score (0-10) for (hop query, chunk) over dense top-10.
Stage train  : group split by question; BCE on score/10 from an ms-marco CE init.
Stage eval   : base CE vs distilled on held-out questions, gap closure@3.
"""
import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .ce_client import CERemote
from .e5_gap import build, measure
from .e6_llm_upper import JUDGE, parse_int
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
_lock = threading.Lock()


def collect(rows, corpus, llm, workers=8, topn=10):
    path = os.path.join(RES, "e7_pairs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["qid"], r["hop"], r["idx"]))
    jobs = []
    for r in rows:
        top = list(np.argsort(-r["dense_scores"])[:topn])
        gold = set(__import__("src.e5_gap", fromlist=["_GOLD"])._GOLD.get((r["qid"], r["hop"]), []))
        for idx in top:
            if (r["qid"], r["hop"], idx) in done:
                continue
            jobs.append({"qid": r["qid"], "hop": int(r["hop"]), "idx": int(idx),
                         "q": r["text"], "p": corpus[idx][:1200],
                         "is_gold": bool(idx in gold),
                         "dense_rank": None if r["rank_dense"] is None else int(r["rank_dense"])})
    print(f"[e7] {len(jobs)} pairs to judge ({len(done)} cached)")

    def work(j):
        try:
            s = parse_int(llm.chat([{"role": "user", "content": JUDGE.format(q=j["q"], p=j["p"])}],
                                   max_tokens=8))
        except Exception:
            s = 0
        return {**j, "score": s}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            rec = fut.result()
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if i % 200 == 0 or i == len(jobs):
                print(f"[e7] {i}/{len(jobs)} calls={llm.calls}", flush=True)


class PairDS(Dataset):
    def __init__(self, recs, tok, max_len=192):
        self.recs, self.tok, self.max_len = recs, tok, max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        b = self.tok(r["q"], r["p"], truncation=True, max_length=self.max_len,
                     padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in b.items()}, torch.tensor(r["score"] / 10.0, dtype=torch.float)


def train(seed=0, epochs=2, lr=2e-5, batch=32, train_frac=0.7):
    recs = [json.loads(l) for l in open(os.path.join(RES, "e7_pairs.jsonl"), encoding="utf-8") if l.strip()]
    qids = sorted({r["qid"] for r in recs})
    random.Random(seed).shuffle(qids)
    n_tr = int(len(qids) * train_frac)
    tr_q, te_q = set(qids[:n_tr]), set(qids[n_tr:])
    tr = [r for r in recs if r["qid"] in tr_q]
    te = [r for r in recs if r["qid"] in te_q]
    print(f"[e7] train pairs={len(tr)} ({len(tr_q)} q)  test pairs={len(te)} ({len(te_q)} q)")

    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    torch.set_num_threads(8)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
    dl = DataLoader(PairDS(tr, tok), batch_size=batch, shuffle=True)
    mdl.train()
    for ep in range(epochs):
        tot = 0.0
        for i, (batch_in, y) in enumerate(dl, 1):
            opt.zero_grad()
            logits = mdl(**batch_in).logits.view(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            tot += loss.item()
            if i % 20 == 0:
                print(f"  ep{ep+1} {i}/{len(dl)} loss={tot/i:.4f}", flush=True)
    out = os.path.join(MODELS, "e7_ce")
    os.makedirs(out, exist_ok=True)
    mdl.save_pretrained(out)
    tok.save_pretrained(out)
    with open(os.path.join(RES, "e7_split.json"), "w") as f:
        json.dump({"train_qids": sorted(tr_q), "test_qids": sorted(te_q)}, f)
    print(f"[e7] saved {out}")
    return te, out


def eval_split(te, model_dir):
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tok0 = AutoTokenizer.from_pretrained(name)
    mdl0 = AutoModelForSequenceClassification.from_pretrained(name).eval()
    tok1 = AutoTokenizer.from_pretrained(model_dir)
    mdl1 = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    torch.set_num_threads(8)

    def score(mdl, tok, pairs):
        out = []
        with torch.inference_mode():
            for i in range(0, len(pairs), 64):
                b = tok([p[0] for p in pairs[i:i + 64]], [p[1] for p in pairs[i:i + 64]],
                        padding=True, truncation=True, max_length=192, return_tensors="pt")
                out.extend(mdl(**b).logits.view(-1).tolist())
        return out

    groups = {}
    for r in te:
        groups.setdefault((r["qid"], r["hop"]), []).append(r)
    pairs = [(r["q"], r["p"]) for r in te]
    s0 = score(mdl0, tok0, pairs)
    s1 = score(mdl1, tok1, pairs)
    for i, r in enumerate(te):
        r["base"] = s0[i]
        r["dist"] = s1[i]

    res = {}
    for label, pred in [("base", "base"), ("distilled", "dist")]:
        hit = reach = gap_hit = gap_n = 0
        for k, g in groups.items():
            if not any(x["is_gold"] for x in g):
                continue
            reach += 1
            order = sorted(g, key=lambda x: -x[pred])
            ok = any(x["is_gold"] for x in order[:3])
            hit += ok
            if g[0]["dense_rank"] and g[0]["dense_rank"] > 3:
                gap_n += 1
                gap_hit += ok
        res[label] = {"reachable": reach, "closure@3": hit / reach if reach else None,
                      "gap_cases": gap_n, "gap_closure@3": gap_hit / gap_n if gap_n else None}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(RES, "e7_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["collect", "train", "eval", "all"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    qs = load()[:a.n]
    ce = CERemote()
    rows, corpus, meta, emb = measure(qs, ce)
    import src.e5_gap as M
    M._GOLD = build(qs)[2]

    if a.stage in ("collect", "all"):
        collect(rows, corpus, LLM(os.path.join(ROOT, "cache", "llm.sqlite")), a.workers)
    out = os.path.join(MODELS, "e7_ce")
    if a.stage in ("train", "all"):
        te, out = train()
    if a.stage in ("eval", "all"):
        recs = [json.loads(l) for l in open(os.path.join(RES, "e7_pairs.jsonl"), encoding="utf-8") if l.strip()]
        split = json.load(open(os.path.join(RES, "e7_split.json")))
        te = [r for r in recs if r["qid"] in set(split["test_qids"])]
        eval_split(te, out)
