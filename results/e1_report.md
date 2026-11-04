# E1: coverage objective on natural corpora (MuSiQue)

- questions: 200　|　per-question pool: 19–20 paragraphs　|　pooled: 3999 paragraphs
- natural exact duplicates: 467 texts appear more than once (max 17 copies); repeated titles: 547

## per-question pool: hop coverage@k (oracle labels)

| k | pointwise | greedy coverage | pointwise redundancy |
|---|---|---|---|
| 1 | 0.167 | 0.400 | 0.000 |
| 2 | 0.306 | 0.799 | 0.000 |
| 3 | 0.378 | 0.951 | 0.000 |
| 4 | 0.433 | 1.000 | 0.000 |
| 6 | 0.541 | 1.000 | 0.000 |

## pooled corpus (candidates = BM25 top-50)

- candidate recall (share of hops whose gold is inside top-50): 0.610 — this is the ceiling of any selector on this candidate set

| k | pointwise | greedy coverage | top-k has same-title dup |
|---|---|---|---|
| 1 | 0.208 | 0.371 | 0.000 |
| 2 | 0.256 | 0.581 | 0.315 |
| 3 | 0.288 | 0.610 | 0.485 |
| 4 | 0.315 | 0.610 | 0.580 |
| 6 | 0.350 | 0.610 | 0.675 |

> pointwise redundancy = share of selected docs that cover an already-covered hop.
