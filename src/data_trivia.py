"""Single-hop pool from TriviaQA rc.web: answer-bearing passages + distractors."""
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")


def build(n=200, seed=42, out_name="trivia_200.jsonl"):
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, "trivia.parquet"))
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)

    def passages(ex):
        out = []
        for d in ex["search_results"]["description"]:
            if isinstance(d, str) and d.strip():
                out.append(d.strip())
        for w in ex["entity_pages"]["wiki_context"]:
            if isinstance(w, str) and w.strip():
                out.append(w.strip())
        seen, uniq = set(), []
        for p in out:
            k = p[:120]
            if k not in seen:
                seen.add(k)
                uniq.append(p[:1500])
        return uniq

    pool, meta = [], []
    for i in idx[:1500]:
        for p in passages(df.iloc[i]):
            pool.append(p)
            meta.append(i)

    rows = []
    for i in idx[:1500]:
        if len(rows) >= n:
            break
        ex = df.iloc[i]
        aliases = [str(a).lower().strip() for a in ex["answer"]["aliases"] if str(a).strip()]
        if not aliases:
            continue
        ps = passages(ex)
        gold = [p for p in ps if any(a and a in p.lower() for a in aliases)]
        if not gold:
            continue
        gold_set = set(gold)
        negs = []
        rng = random.Random(seed + len(rows))
        tries = 0
        while len(negs) < 9 and tries < 500:
            j = rng.randrange(len(pool))
            if meta[j] != i and pool[j] not in gold_set and pool[j] not in negs:
                negs.append(pool[j])
            tries += 1
        docs = [{"doc_id": f"trivia{len(rows)}#g{k}", "title": "gold", "text": p,
                 "sentences": [p], "is_gold": True} for k, p in enumerate(gold[:3])]
        docs += [{"doc_id": f"trivia{len(rows)}#n{k}", "title": "distractor", "text": p,
                  "sentences": [p], "is_gold": False} for k, p in enumerate(negs)]
        rng.shuffle(docs)
        rows.append({"qid": f"trivia{len(rows)}", "question": ex["question"],
                     "answer": str(ex["answer"]["value"]), "docs": docs})
    path = os.path.join(RAW, out_name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, rows


if __name__ == "__main__":
    p, rows = build()
    print("wrote", p, len(rows))
    print("gold per q:", [sum(d["is_gold"] for d in r["docs"]) for r in rows[:8]])
    print("example:", rows[0]["question"], "|", rows[0]["answer"])
