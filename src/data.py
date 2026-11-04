"""Build the pilot corpus from HotpotQA distractor validation.

Each question keeps its own 10-paragraph candidate pool (the standard distractor
setting). Gold paragraphs are the titles listed in supporting_facts.
"""
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(DATA_DIR, "raw")

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build(n_questions=20, seed=42, out_name="hotpot20.jsonl"):
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=seed).select(range(n_questions))

    rows = []
    for ex in ds:
        qid = ex["id"]
        docs = []
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            docs.append(
                {
                    "doc_id": f"{qid}#{title}",
                    "title": title,
                    "sentences": sents,
                    "text": " ".join(sents),
                }
            )
        gold = sorted(set(ex["supporting_facts"]["title"]))
        # sentence-level gold, for locating the answer span
        gold_sents = {}
        for t, sid in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
            gold_sents.setdefault(t, set()).add(int(sid))
        for d in docs:
            d["is_gold"] = d["title"] in gold
            d["contains_answer"] = any(
                _answer_in(s, ex["answer"]) for s in d["sentences"]
            )

        rows.append(
            {
                "qid": qid,
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "gold_titles": gold,
                "gold_sents": {k: sorted(v) for k, v in gold_sents.items()},
                "docs": docs,
            }
        )

    path = os.path.join(RAW, out_name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, rows


def _answer_in(sentence, answer):
    a = answer.strip().lower()
    if not a:
        return False
    return re.search(r"\b" + re.escape(a) + r"\b", sentence.lower()) is not None


def load_questions(path=None):
    if path is None:
        path = os.path.join(RAW, "hotpot20.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    path, rows = build()
    n_gold = [sum(d["is_gold"] for d in r["docs"]) for r in rows]
    print(f"wrote {path}: {len(rows)} questions")
    print("gold docs per question:", n_gold)
    print("docs containing answer:", [sum(d["contains_answer"] for d in r["docs"]) for r in rows])
