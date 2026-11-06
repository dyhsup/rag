"""Counterfactual support spans: which sentence in a passage is load-bearing?

For each (requirement, gold passage) we ask the generator for the single
shortest sentence that on its own answers the requirement, then map it back to
a sentence index.  The index is used later to build deletion pairs:

    d            the full passage
    d \\ support   delete the load-bearing sentence   -> answer changes
    d \\ control   delete another sentence            -> answer preserved

A matcher trained on these pairs must put its mass on the support content
instead of on topical overlap.
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.e3_binding import render  # noqa: E402
from src.llm import LLM  # noqa: E402
from src.musique import load  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "dist", "support")

PROMPT = ("Passage:\n{p}\n\nQuestion: {q}\nAnswer: {a}\n\n"
          "Quote the single shortest sentence from the passage that, on its own, "
          "is enough to answer the question. Output only that sentence, verbatim.")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main(start=400, n=600, workers=8):
    qs = load(os.path.join(ROOT, "data", "raw", "musique1000.jsonl"))[start:start + n]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "train.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["key"])
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite"))

    def work(q):
        out = []
        for hi, h in enumerate(q["hops"][:2]):
            gi = next((i for i, d in enumerate(q["docs"][:20]) if d["idx"] == h["para_idx"]), None)
            if gi is None:
                continue
            d = q["docs"][:20][gi]
            req = render(h["question"], [x["answer"] for x in q["hops"][:hi]])
            key = f'{q["qid"]}#h{hi}'
            if key in done:
                continue
            sents = d["sentences"] or [d["text"]]
            try:
                quote = llm.chat([{"role": "user", "content":
                                   PROMPT.replace("{p}", d["text"][:2500]).replace("{q}", req)
                                   .replace("{a}", h["answer"])}], max_tokens=64).strip()
            except Exception:
                quote = ""
            nq = norm(quote)
            sup = None
            if nq:
                sup = max(range(len(sents)), key=lambda i: len(nq) and (
                    len(set(nq.split()) & set(norm(sents[i]).split())) / max(len(set(nq.split())), 1)))
                if len(set(nq.split()) & set(norm(sents[sup]).split())) / max(len(set(nq.split())), 1) < 0.4:
                    sup = None
            if sup is None:
                ans = norm(h["answer"])
                cand = [i for i, s in enumerate(sents) if ans and ans in norm(s)]
                sup = cand[0] if cand else min(range(len(sents)), key=lambda i: -len(norm(sents[i])))
            ctrl = next((i for i in range(len(sents)) if i != sup), None)
            if ctrl is None:
                continue
            out.append({"key": key, "qid": q["qid"], "q": req, "ans": h["answer"],
                        "text": d["text"], "sents": sents, "sup": sup, "ctrl": ctrl})
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, recs in enumerate(ex.map(work, qs), 1):
            with open(path, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"[support] {i}/{len(qs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=400)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.start, a.n, a.workers)
