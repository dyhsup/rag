"""E11: build pairwise preference data with an LLM (A/B, position-swapped).

For each training hop:
  gap hop  (gold dense_rank > 3): gold vs 3 hardest negatives
  easy hop (gold in top-3)      : gold vs 2 hardest negatives
The LLM answers "A", "B" or "TIE"; ties are dropped.
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import LLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_lock = threading.Lock()

PROMPT = """Question: {q}
Passage A: {a}
Passage B: {b}

Which passage better supports answering the question? Reply with exactly one word: A, B, or TIE.
Answer:"""


def split_train():
    recs = [json.loads(l) for l in open(os.path.join(RES, "e10_scored.jsonl"), encoding="utf-8") if l.strip()]
    qids = sorted({r["qid"] for r in recs})
    random.Random(0).shuffle(qids)
    n = int(len(qids) * 0.7)
    return set(qids[:n])


def build_pairs():
    recs = [json.loads(l) for l in open(os.path.join(RES, "e10_scored.jsonl"), encoding="utf-8") if l.strip()]
    tr = split_train()
    by = defaultdict(list)
    for r in recs:
        if r["qid"] in tr:
            by[(r["qid"], r["hop"])].append(r)
    pairs = []
    for key, items in by.items():
        golds = [x for x in items if x["label"] == 1]
        negs = sorted([x for x in items if x["label"] == 0], key=lambda x: x["dense_rank"])
        if not golds or not negs:
            continue
        gold = min(golds, key=lambda x: x["dense_rank"])
        k = 3 if gold["dense_rank"] > 3 else 2
        for neg in negs[:k]:
            h = int(hashlib.sha1(f"{key}:{neg['idx']}".encode()).hexdigest()[:8], 16)
            swap = h % 2 == 1
            a, b = (neg, gold) if swap else (gold, neg)
            pairs.append({"qid": key[0], "hop": int(key[1]),
                          "q": gold["q"], "text_a": a["p"], "text_b": b["p"],
                          "gold_is_a": 0 if swap else 1})
    return pairs


def parse_choice(t):
    t = (t or "").strip().upper()
    m = re.match(r"^(A|B|TIE)\b", t)
    if m:
        return m.group(1)
    for c in ["TIE", "A", "B"]:
        if c in t:
            return c
    return "TIE"


def main(workers=12):
    path = os.path.join(RES, "e11_pairs.jsonl")
    pairs = build_pairs()
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["qid"], r["hop"], r["i"]))
    for i, p in enumerate(pairs):
        p["i"] = i
    todo = [p for p in pairs if (p["qid"], p["hop"], p["i"]) not in done]
    print(f"[e11] pairs={len(pairs)} to_label={len(todo)}")
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def work(p):
        try:
            raw = llm.chat([{"role": "user", "content": PROMPT.format(
                q=p["q"], a=p["text_a"][:1200], b=p["text_b"][:1200])}], max_tokens=8)
            c = parse_choice(raw)
        except Exception:
            c = "TIE"
        if c == "TIE":
            pref = None
        elif c == "A":
            pref = 1 if p["gold_is_a"] else 0
        else:
            pref = 0 if p["gold_is_a"] else 1
        return {**p, "choice": c, "pref": pref, "raw": c}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(work, p) for p in todo]), 1):
            rec = fut.result()
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if k % 500 == 0 or k == len(todo):
                print(f"[e11] {k}/{len(todo)} calls={llm.calls}", flush=True)

    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    seen = {}
    for r in recs:
        seen[(r["qid"], r["hop"], r["i"])] = r
    recs = list(seen.values())
    tie = sum(1 for r in recs if r["pref"] is None)
    pos = sum(1 for r in recs if r["pref"] == 1)
    print(json.dumps({"n": len(recs), "gold_preferred": pos,
                      "neg_preferred": len(recs) - tie - pos, "tie": tie,
                      "gold_win_rate": pos / max(1, len(recs) - tie)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    main(ap.parse_args().workers)
