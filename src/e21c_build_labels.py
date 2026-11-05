"""Build counterfactual marginal-utility labels for CMU distillation."""
import argparse, json, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "dist", "cmu")
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


def main(start=200, n=400, workers=8):
    qs = load(os.path.join(ROOT, "data", "raw", "musique1000.jsonl"))[start:start + n]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "train_labels.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[cmu-labels] {len(todo)} questions")

    def work(q):
        docs = [d["text"] for d in q["docs"]][:20]
        base = logp(q["question"], [], q["answer"])
        sel, cur = [], base
        rows = []
        for _ in range(2):
            vals = []
            for i, d in enumerate(docs):
                if i in sel:
                    continue
                v = logp(q["question"], [docs[j] for j in sel] + [d], q["answer"])
                vals.append((v, i))
            if not vals:
                break
            for v, i in vals:
                rows.append({"qid": q["qid"], "q": q["question"],
                             "sel": [docs[j] for j in sel], "cand": docs[i],
                             "delta": v - cur, "label": 1 if i == max(vals)[1] else 0})
            v, i = max(vals)
            sel.append(i); cur = v
        return {"qid": q["qid"], "rows": rows}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, q) for q in todo]), 1):
            rec = fut.result()
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    for r in rec["rows"]:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"[cmu-labels] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    main(a.start, a.n)
