"""E12: forced A/B pairwise labels (no TIE option)."""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import LLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_lock = threading.Lock()

PROMPT = """Question: {q}
Passage A: {a}
Passage B: {b}

Which passage better supports answering the question? You must choose one.
Reply with exactly one letter: A or B.
Answer:"""


def parse(t):
    t = (t or "").strip().upper()
    m = re.match(r"^[^AB]{0,8}(A|B)\b", t)
    if m:
        return m.group(1)
    if "A" in t and "B" not in t:
        return "A"
    if "B" in t and "A" not in t:
        return "B"
    return None


def main(workers=12):
    src = os.path.join(RES, "e11_pairs.jsonl")
    out = os.path.join(RES, "e12_forced.jsonl")
    recs = {}
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        recs[(r["qid"], r["hop"], r["i"])] = r
    recs = list(recs.values())
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["qid"], r["hop"], r["i"]))
    todo = [r for r in recs if (r["qid"], r["hop"], r["i"]) not in done]
    print(f"[e12] pairs={len(recs)} to_label={len(todo)}")
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def work(r):
        choice = None
        for _ in range(2):
            try:
                raw = llm.chat([{"role": "user", "content": PROMPT.format(
                    q=r["q"], a=r["text_a"][:1200], b=r["text_b"][:1200])}], max_tokens=8)
                choice = parse(raw)
            except Exception:
                choice = None
            if choice:
                break
        if choice is None:
            pref = None
        elif choice == "A":
            pref = 1 if r["gold_is_a"] else 0
        else:
            pref = 0 if r["gold_is_a"] else 1
        return {**r, "choice2": choice, "pref2": pref}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(work, r) for r in todo]), 1):
            rec = fut.result()
            with _lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if k % 1000 == 0 or k == len(todo):
                print(f"[e12] {k}/{len(todo)} calls={llm.calls}", flush=True)

    recs = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    seen = {(r["qid"], r["hop"], r["i"]): r for r in recs}
    recs = list(seen.values())
    pos = sum(1 for r in recs if r["pref2"] == 1)
    neg = sum(1 for r in recs if r["pref2"] == 0)
    print(json.dumps({"n": len(recs), "gold_preferred": pos, "neg_preferred": neg,
                      "invalid": len(recs) - pos - neg,
                      "gold_win_rate": pos / max(1, pos + neg)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    main(ap.parse_args().workers)
