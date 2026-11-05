"""QuALITY long-document MCQ: article -> 10 chunks, options, gold option."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")


def split_k(text, k=10):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []
    total = sum(len(p) for p in paras)
    target = max(1, total // k)
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > target and len(out) < k - 1:
            out.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip()
    if cur:
        out.append(cur)
    return out


def build(n=200, seed=42, out_name="quality_200.jsonl"):
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, "quality.parquet"))
    rows = []
    for i in range(min(len(df), n * 2)):
        ex = df.iloc[i]
        opts = [str(o) for o in ex["options"]]
        gi = int(ex["answer"])
        if gi >= len(opts):
            continue
        gold = opts[gi]
        chunks = split_k(str(ex["article"]), 10)
        docs = []
        for j, c in enumerate(chunks):
            docs.append({"doc_id": f"q{i}#{j}", "title": "chunk", "text": c,
                         "sentences": [c], "is_gold": gold.lower() in c.lower()})
        rows.append({"qid": f"q{i}", "question": str(ex["question"]), "answer": gold,
                     "options": opts, "docs": docs})
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
    print("example:", rows[0]["question"][:80], "|", rows[0]["options"], "|", rows[0]["answer"])
