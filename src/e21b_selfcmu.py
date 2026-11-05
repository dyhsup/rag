"""Legitimate CMU: use the model's OWN answer (no gold) as the utility target."""
import argparse, json, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .ce_client import CERemote
from .e3_binding import em, f1
from .e4_pipeline import READ, fmt
from .llm import LLM
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
EPS = ["http://10.94.199.216:8000/v1", "http://10.94.199.217:8000/v1"]
MODEL = "qwen3.6-27b"
_lock = threading.Lock(); _rr = [0]


def logp(q, passages, answer):
    body = "\n".join(f"[{i+1}] {p[:400]}" for i, p in enumerate(passages)) or "(none)"
    prompt = f"Question: {q}\nPassages:\n{body}\nAnswer: {answer}"
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 0, "echo": True,
               "logprobs": 0, "temperature": 0}
    for _ in range(3):
        with _lock:
            ep = EPS[_rr[0] % len(EPS)]; _rr[0] += 1
        try:
            r = requests.post(ep + "/completions", json=payload, proxies={}, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(r.text[:100])
            lp = r.json()["choices"][0]["logprobs"]
            cut = prompt.rfind("Answer: ") + len("Answer: ")
            return sum(t for o, t in zip(lp["text_offset"], lp["token_logprobs"])
                       if o >= cut and t is not None)
        except Exception:
            continue
    return float("-inf")


def main(n=100, workers=6, k=2):
    qs = load()[:n]
    ce = CERemote(); llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))
    path = os.path.join(RES, "e21b_selfcmu.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[e21b] {len(todo)}")

    def work(q):
        docs = [d["text"] for d in q["docs"]][:20]
        try:
            csc = ce.rerank(q["question"], docs)
            ce_idx = sorted(range(len(docs)), key=lambda i: -csc[i])[:5]
        except Exception:
            ce_idx = list(range(5))
        # 1) self answer from CE top-5
        try:
            a0 = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt([docs[i] for i in ce_idx]))}],
                          max_tokens=32).strip().split("\n")[0]
        except Exception:
            a0 = ""
        # 2) CMU on the self answer
        sel = []
        if a0 and "not contain" not in a0.lower() and "cannot" not in a0.lower():
            cur = logp(q["question"], [], a0)
            for _ in range(k):
                best, bd, blp = -1e9, None, cur
                for i, d in enumerate(docs):
                    if i in sel:
                        continue
                    v = logp(q["question"], [docs[j] for j in sel] + [d], a0)
                    if v - cur > best:
                        best, bd, blp = v - cur, i, v
                if bd is None:
                    break
                sel.append(bd); cur = blp
        # 3) read final from the selected set (fallback to CE top-5)
        ps = [docs[i] for i in (sel or ce_idx)]
        try:
            pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt(ps))}],
                            max_tokens=32).strip().split("\n")[0]
        except Exception:
            pred = ""
        rec = {"qid": q["qid"], "self_answer": a0, "sel": sel,
               "pred": pred, "gold": q["answer"],
               "em": em(pred, q["answer"]), "f1": f1(pred, q["answer"]),
               "self_em": em(a0, q["answer"]), "ce_sel": ce_idx}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            fut.result()
            if i % 20 == 0 or i == len(todo):
                print(f"[e21b] {i}/{len(todo)}", flush=True)

    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(json.dumps({"n": len(recs),
                      "self_answer_EM": sum(r["self_em"] for r in recs) / len(recs),
                      "final_EM": sum(r["em"] for r in recs) / len(recs),
                      "changed_selection": sum(1 for r in recs if r["sel"] and set(r["sel"]) != set(r["ce_sel"][:2])) / len(recs)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    main(a.n)
