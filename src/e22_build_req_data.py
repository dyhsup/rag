"""Build requirement-conditioned reranking data (RQ vs QQ control).

For every training question and every hop we emit one positive (the paragraph
that supports that hop) and several negatives drawn from the same 20-paragraph
pool.  The other hops' supporting paragraphs are deliberately kept as negatives:
a model that only learns "this paragraph is about the question" cannot reject
them, only a model that binds to the specific requirement can.

Two files are written with *identical* pairs and labels:
  data_hop.jsonl  query = the hop sub-question   (requirement-conditioned)
  data_q.jsonl    query = the full question      (control)
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.e3_binding import render  # noqa: E402
from src.musique import load  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "dist", "reqce")
STOP = set("a an the of to in on at for and or is are was were be been by with as that this these those it its from who what which when where how did does do".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def main(start=200, n=400, neg_hard=4, neg_rand=2, seed=0):
    qs = load(os.path.join(ROOT, "data", "raw", "musique1000.jsonl"))[start:start + n]
    rnd = random.Random(seed)
    hop_rows, q_rows = [], []
    for q in qs:
        docs = q["docs"][:20]
        text = [d["text"] for d in docs]
        gold = {i for i, d in enumerate(docs) if d["is_gold"]}
        for h_i, hop in enumerate(q["hops"][:2]):
            pos_i = next((i for i, d in enumerate(docs) if d["idx"] == hop["para_idx"]), None)
            if pos_i is None:
                continue
            bound = " ".join(render(hop["question"], [h["answer"] for h in q["hops"][:h_i]]).split())
            rtok = toks(bound)
            cands = [i for i in range(len(docs)) if i != pos_i]
            hard = sorted(cands, key=lambda i: -len(rtok & toks(text[i])))[:neg_hard]
            rest = [i for i in cands if i not in hard]
            negs = hard + rnd.sample(rest, min(neg_rand, len(rest)))
            qid = f'{q["qid"]}#h{h_i}'
            hop_rows.append({"qid": qid, "q": bound, "p": text[pos_i], "label": 1})
            q_rows.append({"qid": qid, "q": q["question"], "p": text[pos_i], "label": 1})
            for i in negs:
                hop_rows.append({"qid": qid, "q": bound, "p": text[i], "label": 0})
                q_rows.append({"qid": qid, "q": q["question"], "p": text[i], "label": 0})
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("data_hop.jsonl", hop_rows), ("data_q.jsonl", q_rows)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hop_rows={len(hop_rows)} q_rows={len(q_rows)} written to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--neg-hard", type=int, default=4)
    ap.add_argument("--neg-rand", type=int, default=2)
    a = ap.parse_args()
    main(a.start, a.n, a.neg_hard, a.neg_rand)
