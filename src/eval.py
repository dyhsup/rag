"""Retrieval and correlation metrics for the pilot."""
import math
import random


def _order(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def mrr_at_k(scores, gold, k=10):
    for r, i in enumerate(_order(scores)[:k], 1):
        if i in gold:
            return 1.0 / r
    return 0.0


def ndcg_at_k(scores, gold, k=10):
    order = _order(scores)[:k]
    dcg = sum(1.0 / math.log2(r + 1) for r, i in enumerate(order, 1) if i in gold)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(scores, gold, k=2):
    top = set(_order(scores)[:k])
    return len(top & set(gold)) / max(1, len(gold))


def auc(scores, labels):
    """Rank-based AUC; labels are 0/1. None if a class is missing."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values, n_boot=1000, seed=0):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return mean(values), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def paired_bootstrap(a, b, n_boot=1000, seed=0):
    """Mean of (a - b) with 95% percentile CI over questions."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    diffs = [x - y for x, y in pairs]
    means = []
    for _ in range(n_boot):
        s = [rng.choice(diffs) for _ in diffs]
        means.append(sum(s) / len(s))
    means.sort()
    return sum(diffs) / len(diffs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
