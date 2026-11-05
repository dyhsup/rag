"""Single-hop pool from SQuAD v1 validation: gold context + 9 distractors."""
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")


def build(n=200, seed=42, out_name="squad_200.jsonl"):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    chosen = idx[:n]
    pool = [(ds[i]["context"], ds[i]["title"]) for i in idx]
    rows = []
    for j, i in enumerate(chosen):
        ex = ds[i]
        gold_ctx, title = ex["context"], ex["title"]
        negs = []
        tries = 0
        while len(negs) < 9 and tries < 500:
            c, t = pool[random.Random(seed + j * 1000 + tries).randrange(len(pool))]
            tries += 1
            if c != gold_ctx and c not in negs:
                negs.append(c)
        docs = [{"doc_id": f"squad{j}#gold", "title": title, "text": gold_ctx,
                 "sentences": [gold_ctx], "is_gold": True}]
        docs += [{"doc_id": f"squad{j}#neg{k}", "title": "distractor", "text": c,
                  "sentences": [c], "is_gold": False} for k, c in enumerate(negs)]
        random.Random(seed + j).shuffle(docs)
        rows.append({"qid": f"squad{j}", "question": ex["question"],
                     "answer": ex["answers"]["text"][0], "docs": docs})
    path = os.path.join(RAW, out_name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, rows


if __name__ == "__main__":
    p, rows = build()
    print("wrote", p, len(rows))
    print("example:", rows[0]["question"], "|", rows[0]["answer"])
