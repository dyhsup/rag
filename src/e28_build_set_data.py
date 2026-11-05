"""Build set-level contrastive data: a correct evidence pair vs the same pair
with one slot replaced by a topically near-identical passage.

The negative that matters is "topically right, binding wrong": we replace one
gold paragraph with a non-gold paragraph that is lexically the closest to the
question, so a topical matcher cannot tell the two sets apart.
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.musique import load  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "dist", "setce")
STOP = set("a an the of to in on at for and or is are was were be been by with as that this these those it its from who what which when where how did does do".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def main(start=400, n=600, n_neg=3, seed=0):
    qs = load(os.path.join(ROOT, "data", "raw", "musique1000.jsonl"))[start:start + n]
    rnd = random.Random(seed)
    rows = []
    for q in qs:
        docs = q["docs"][:20]
        text = [d["text"] for d in docs]
        gold = [i for i, d in enumerate(docs) if d["is_gold"]]
        if len(gold) < 2:
            continue
        g0, g1 = gold[0], gold[1]
        qt = toks(q["question"])
        cand = [i for i in range(len(docs)) if i not in gold]
        hard = sorted(cand, key=lambda i: -len(qt & toks(text[i])))[:n_neg]
        rows.append({"qid": f'{q["qid"]}#pos', "q": q["question"],
                     "d1": text[g0], "d2": text[g1], "label": 1})
        for i in hard:
            rows.append({"qid": f'{q["qid"]}#n0-{i}', "q": q["question"],
                         "d1": text[g0], "d2": text[i], "label": 0})
            rows.append({"qid": f'{q["qid"]}#n1-{i}', "q": q["question"],
                         "d1": text[i], "d2": text[g1], "label": 0})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rows={len(rows)} pos={sum(r['label'] for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=400)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-neg", type=int, default=3)
    a = ap.parse_args()
    main(a.start, a.n, a.n_neg)
