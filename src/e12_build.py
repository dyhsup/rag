"""Build A (GT pointwise) and B (GT pairwise) training files."""
import json
import os
import random
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "dist", "e12")


def split_train(recs):
    qids = sorted({r["qid"] for r in recs})
    random.Random(0).shuffle(qids)
    return set(qids[:int(len(qids) * 0.7)])


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(RES, "e10_scored.jsonl"), encoding="utf-8") if l.strip()]
    with open(os.path.join(OUT, "pairs_gt_pointwise.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"qid": r["qid"], "hop": r["hop"], "idx": r["idx"],
                                "q": r["q"], "p": r["p"], "label": r["label"],
                                "dense_rank": r["dense_rank"], "gold_rank": r["gold_rank"],
                                "base_logit": r["base_logit"]}, ensure_ascii=False) + "\n")

    tr = split_train(recs)
    by = defaultdict(list)
    for r in recs:
        if r["qid"] in tr:
            by[(r["qid"], r["hop"])].append(r)
    n = 0
    with open(os.path.join(OUT, "pairs_gt_pairwise.jsonl"), "w", encoding="utf-8") as f:
        for (qid, hop), items in by.items():
            golds = [x for x in items if x["label"] == 1]
            negs = sorted([x for x in items if x["label"] == 0], key=lambda x: x["dense_rank"])
            if not golds or not negs:
                continue
            gold = min(golds, key=lambda x: x["dense_rank"])
            k = 3 if gold["dense_rank"] > 3 else 2
            for neg in negs[:k]:
                f.write(json.dumps({"qid": qid, "hop": int(hop), "q": gold["q"],
                                    "text_a": gold["p"], "text_b": neg["p"], "pref": 1},
                                   ensure_ascii=False) + "\n")
                n += 1
    print("pointwise", len(recs), "gt_pairwise", n)


if __name__ == "__main__":
    main()
