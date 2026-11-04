"""Build a MuSiQue-compatible cache from 2WikiMultihopQA dev.parquet.

evidences = [(head, relation, tail), ...] lets us reconstruct hop queries and
per-hop gold answers; supporting_facts gives the gold paragraphs.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")


def build(n=200, seed=42, out_name="wiki2_200.jsonl"):
    import pandas as pd

    df = pd.read_parquet(os.path.join(RAW, "2wiki_dev.parquet"))
    if seed is not None:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows = []
    for _, ex in df.head(n * 3).iterrows():
        def parse(v):
            return json.loads(v) if isinstance(v, str) else v
        ctx = parse(ex["context"])
        if not isinstance(ctx, (list, tuple)):
            continue
        docs = []
        for title, sents in ctx:
            sents = list(sents) if isinstance(sents, (list, tuple)) else [str(sents)]
            docs.append({"doc_id": f'{ex["_id"]}#{title}', "title": title,
                         "text": " ".join(sents), "sentences": sents,
                         "is_gold": False})
        gold_titles = {t for t, _ in parse(ex["supporting_facts"])}
        for d in docs:
            d["is_gold"] = d["title"] in gold_titles
        ev = parse(ex["evidences"])
        support_titles = [t for t, _ in parse(ex["supporting_facts"])]

        def find_doc(name, pos):
            key = name.lower().strip()
            for j, d in enumerate(docs):
                t = d["title"].lower().strip()
                if t == key or t.startswith(key) or key in t:
                    return j
            for j, d in enumerate(docs):
                if pos < len(support_titles) and d["title"] == support_titles[pos]:
                    return j
            return -1

        hops = []
        for i, (head, rel, tail) in enumerate(ev):
            idx = find_doc(head, i)
            hops.append({"hop": i, "question": f"{head} {rel} ?",
                         "answer": str(tail), "para_idx": idx if idx is not None else -1})
        if not hops or any(h["para_idx"] < 0 for h in hops):
            continue
        rows.append({"qid": str(ex["_id"]), "question": ex["question"],
                     "answer": str(ex["answer"]), "n_hops": len(hops),
                     "hops": hops, "docs": docs, "type": ex.get("type", "")})
        if len(rows) >= n:
            break
    path = os.path.join(RAW, out_name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, rows


if __name__ == "__main__":
    p, rows = build()
    print("wrote", p, len(rows))
    print("hop counts:", {k: sum(1 for r in rows if r["n_hops"] == k) for k in (2, 3, 4)})
    print("types:", {t: sum(1 for r in rows if r["type"] == t) for t in {r["type"] for r in rows}})
    r = rows[0]
    print("example:", r["question"], "|", [h["question"] for h in r["hops"]], "|", r["answer"])
