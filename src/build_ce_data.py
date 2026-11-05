"""Build per-dataset CE training data (10-passage pools, grouped by question).

Output: dist/ce_data/<dataset>.jsonl  with {qid, q, p, label, dataset}
Training questions are disjoint from the evaluation sets used in the paper.
"""
import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "dist", "ce_data")
CHUNK = 40
STRIDE = 20


def chunks(text, width=CHUNK, stride=STRIDE):
    toks = text.split()
    if len(toks) <= width:
        return [" ".join(toks)] if len(toks) >= 8 else []
    out = []
    for i in range(0, len(toks) - width + 1, stride):
        out.append(" ".join(toks[i:i + width]))
    if (len(toks) - width) % stride:
        out.append(" ".join(toks[-width:]))
    return out


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def write_rows(name, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    qids = len({r["qid"] for r in rows})
    print(f"[ce-data] {name}: pairs={len(rows)} questions={qids} -> {path}")


def build_musique(n_pairs=12000):
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "results", "e10_scored.jsonl"), encoding="utf-8")]
    eval_qids = {json.loads(l)["qid"] for l in open(os.path.join(RAW, "musique200.jsonl"), encoding="utf-8")}
    by = {}
    for r in recs:
        if r["qid"] in eval_qids:
            continue
        by.setdefault((r["qid"], r["hop"]), []).append(r)
    rows = []
    for (qid, hop), items in by.items():
        pos = [x for x in items if x["label"] == 1][:2]
        neg = [x for x in items if x["label"] == 0][:8]
        for x in pos:
            rows.append({"qid": f"{qid}#{hop}", "q": x["q"], "p": x["p"], "label": 1})
        for x in neg:
            rows.append({"qid": f"{qid}#{hop}", "q": x["q"], "p": x["p"], "label": 0})
        if len(rows) >= n_pairs:
            break
    write_rows("musique", rows[:n_pairs])


def build_pool_dataset(name, questions, n_questions=600, n_neg=9, seed=42):
    """questions: list of dicts {qid, question, gold_texts, distractor_texts}"""
    rng = random.Random(seed)
    rng.shuffle(questions)
    rows = []
    for q in questions[:n_questions]:
        golds = [t for t in q["gold_texts"] if t][:3]
        negs = [t for t in q["distractor_texts"] if t][:n_neg]
        if not golds or len(negs) < 3:
            continue
        for t in golds:
            rows.append({"qid": q["qid"], "q": q["question"], "p": t, "label": 1})
        for t in negs:
            rows.append({"qid": q["qid"], "q": q["question"], "p": t, "label": 0})
    write_rows(name, rows)


def from_squad(n=600):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="train")
    idx = list(range(len(ds)))
    random.Random(42).shuffle(idx)
    contexts = [ds[i]["context"] for i in idx]
    qs = []
    for j, i in enumerate(idx[:n * 3]):
        ex = ds[i]
        qs.append({"qid": f"squad_tr{i}", "question": ex["question"],
                   "gold_texts": [ex["context"]],
                   "distractor_texts": [contexts[(j * 7 + k) % len(contexts)] for k in range(1, 10)]})
        if len(qs) >= n:
            break
    build_pool_dataset("squad", qs, n)


def from_trivia(n=600):
    import pandas as pd
    path = os.path.join(RAW, "trivia_train.parquet")
    if not os.path.exists(path):
        import requests
        u = ("https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/main/"
             "rc.web/train-00000-of-00019.parquet")
        open(path, "wb").write(requests.get(u, timeout=900).content)
    df = pd.read_parquet(path)
    idx = list(range(len(df)))
    random.Random(42).shuffle(idx)
    pool = []
    for i in idx:
        for d in df.iloc[i]["search_results"]["description"]:
            if isinstance(d, str) and d.strip():
                pool.append(d.strip()[:1500])
    qs = []
    for j, i in enumerate(idx):
        ex = df.iloc[i]
        aliases = [str(a).lower() for a in ex["answer"]["aliases"]]
        ps = [d.strip()[:1500] for d in ex["search_results"]["description"] if isinstance(d, str) and d.strip()]
        gold = [p for p in ps if any(a in p.lower() for a in aliases)]
        if not gold:
            continue
        negs, t = [], 0
        while len(negs) < 9 and t < 300:
            p = pool[(j * 13 + t * 17) % len(pool)]
            if p not in gold and p not in negs:
                negs.append(p)
            t += 1
        qs.append({"qid": f"trivia_tr{i}", "question": ex["question"],
                   "gold_texts": gold, "distractor_texts": negs})
        if len(qs) >= n:
            break
    build_pool_dataset("trivia", qs, n)


def from_2wiki(n=600):
    import pandas as pd
    path = os.path.join(RAW, "2wiki_train.parquet")
    if not os.path.exists(path):
        import requests
        u = ("https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/train.parquet")
        open(path, "wb").write(requests.get(u, timeout=900).content)
    df = pd.read_parquet(path)
    idx = list(range(len(df)))
    random.Random(42).shuffle(idx)
    qs = []
    for i in idx:
        ex = df.iloc[i]
        parse = lambda v: json.loads(v) if isinstance(v, str) else v
        ctx = parse(ex["context"])
        gold_titles = {t for t, _ in parse(ex["supporting_facts"])}
        golds = [" ".join(s) for t, s in ctx if t in gold_titles]
        negs = [" ".join(s) for t, s in ctx if t not in gold_titles][:9]
        if not golds or len(negs) < 3:
            continue
        qs.append({"qid": f"wiki2_tr{ex['_id']}", "question": ex["question"],
                   "gold_texts": golds[:2], "distractor_texts": negs})
        if len(qs) >= n:
            break
    build_pool_dataset("wiki2", qs, n)


def from_hotpot(n=600):
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="train")
    idx = list(range(len(ds)))
    random.Random(42).shuffle(idx)
    qs = []
    for i in idx:
        ex = ds[i]
        gold_titles = set(ex["supporting_facts"]["title"])
        docs = list(zip(ex["context"]["title"], ex["context"]["sentences"]))
        golds = [" ".join(s) for t, s in docs if t in gold_titles]
        negs = [" ".join(s) for t, s in docs if t not in gold_titles][:9]
        if not golds or len(negs) < 3:
            continue
        qs.append({"qid": f"hotpot_tr{ex['id']}", "question": ex["question"],
                   "gold_texts": golds[:2], "distractor_texts": negs})
        if len(qs) >= n:
            break
    build_pool_dataset("hotpot", qs, n)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_musique()
    from_squad()
    from_trivia()
    from_2wiki()
    from_hotpot()
