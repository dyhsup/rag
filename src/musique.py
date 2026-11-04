"""Load MuSiQue (validation) and cache a compact jsonl with per-hop labels.

MuSiQue's question_decomposition gives paragraph_support_idx for every hop,
which is exactly the supervision a coverage objective needs.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def build(n=200, seed=42, out_name="musique200.jsonl"):
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    rows = []
    for ex in ds:
        paras = ex["paragraphs"]
        decomp = ex["question_decomposition"]

        # hop -> paragraph idx (a paragraph can support several hops)
        hops = []
        for h, step in enumerate(decomp):
            idx = step.get("paragraph_support_idx")
            if idx is None:
                continue
            hops.append({"hop": h, "question": step["question"], "answer": step["answer"], "para_idx": int(idx)})
        if not hops:
            continue

        docs = []
        for p in paras:
            docs.append({
                "doc_id": f'{ex["id"]}#{p["idx"]}',
                "idx": int(p["idx"]),
                "title": p["title"],
                "text": p["paragraph_text"],
                "sentences": _sents(p["paragraph_text"]),
                "is_gold": bool(p.get("is_supporting")),
            })
        rows.append({
            "qid": str(ex["id"]),
            "question": ex["question"],
            "answer": ex["answer"],
            "n_hops": len(hops),
            "hops": hops,
            "docs": docs,
        })

    path = os.path.join(DATA_DIR, out_name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, rows


def _sents(text):
    import re

    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def load(path=None):
    path = path or os.path.join(DATA_DIR, "musique200.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    path, rows = build()
    print(f"wrote {path}: {len(rows)} questions")
    print("hops per question:", {n: sum(1 for r in rows if r["n_hops"] == n) for n in range(1, 6)})
    print("docs per question:", sorted({len(r["docs"]) for r in rows}))
