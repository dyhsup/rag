"""Score the mined E8 pairs with the LLM gap judge."""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .e6_llm_upper import JUDGE, parse_int
from .llm import LLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_lock = threading.Lock()


def main(workers=8):
    src = os.path.join(RES, "e8_mined.jsonl")
    out = os.path.join(RES, "e8_labeled.jsonl")
    recs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["qid"], r["hop"], r["idx"]))
    todo = [r for r in recs if (r["qid"], r["hop"], r["idx"]) not in done]
    print(f"[e8] {len(todo)} to label ({len(done)} cached)")
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def work(r):
        try:
            s = parse_int(llm.chat([{"role": "user", "content": JUDGE.format(q=r["q"], p=r["p"])}],
                                   max_tokens=8))
        except Exception:
            s = 0
        return {**r, "score": s}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, r) for r in todo]), 1):
            rec = fut.result()
            with _lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if i % 200 == 0 or i == len(todo):
                print(f"[e8] {i}/{len(todo)} calls={llm.calls}", flush=True)

    recs = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    import collections
    by_kind = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        by_kind[r["kind"]][0] += 1
        by_kind[r["kind"]][1] += int(r["score"] >= 8)
    print(f"total={len(recs)}")
    for k, (n, pos) in sorted(by_kind.items()):
        print(f"  {k:9s} n={n:5d} teacher_pos={pos:4d} ({pos/n:.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.workers)
