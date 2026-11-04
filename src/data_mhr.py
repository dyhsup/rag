"""Build a 10-passage pool from MultiHopRAG (news domain)."""
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")


def build(n=200, seed=42, out_name="mhr_200.jsonl"):
    path = os.path.join(RAW, "MultiHopRAG.json")
    if not os.path.exists(path):
        import requests
        r = requests.get("https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json",
                         timeout=300)
        open(path, "wb").write(r.content)
    data = json.load(open(path, encoding="utf-8"))
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    chosen = idx[:n]
    pool = []
    for i in chosen:
        for e in data[i]["evidence_list"]:
            pool.append((e["title"], e["fact"]))

    rows = []
    for i in chosen:
        ex = data[i]
        gold = [(e["title"], e["fact"]) for e in ex["evidence_list"]]
        gold_titles = {t for t, _ in gold}
        negs = []
        tries = 0
        while len(negs) < 8 and tries < 200:
            t, f = pool[rng.randrange(len(pool))]
            tries += 1
            if t not in gold_titles and (t, f) not in negs:
                negs.append((t, f))
        docs = [{"doc_id": f"mhr{i}#{k}", "title": t, "text": f, "sentences": [f],
                 "is_gold": True} for k, (t, f) in enumerate(gold)]
        docs += [{"doc_id": f"mhr{i}#neg{k}", "title": t, "text": f, "sentences": [f],
                  "is_gold": False} for k, (t, f) in enumerate(negs)]
        rng.shuffle(docs)
        rows.append({"qid": f"mhr{i}", "question": ex["query"], "answer": ex["answer"],
                     "docs": docs, "type": ex.get("question_type", "")})
    path = os.path.join(RAW, out_name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, rows


if __name__ == "__main__":
    p, rows = build()
    print("wrote", p, len(rows))
    print("gold per q:", [sum(d["is_gold"] for d in r["docs"]) for r in rows[:10]])
    r = rows[0]
    print("example:", r["question"][:100], "|", r["answer"])
