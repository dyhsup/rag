"""A-line scoring: aggregating per-claim evidence scores over a passage.

The hypothesis under test: whole-query similarity rewards topical overlap,
while coverage over atomic claims rewards evidence sufficiency.
"""
from .claims import claim_query_text


def _minmax(arr):
    lo, hi = min(arr), max(arr)
    if hi - lo < 1e-9:
        return [0.5] * len(arr)
    return [(x - lo) / (hi - lo) for x in arr]


def score_variants(index, question, claims, cover_tau=0.5):
    """Return {variant_name: [score per doc]}."""
    n_docs = len(index.docs)
    whole = _minmax(index.score_docs(question))

    per_claim = []
    for c in claims:
        sent_scores = index.score_sentences(claim_query_text(c))
        doc_scores = index.sentence_scores_to_docs(sent_scores)
        per_claim.append(_minmax(doc_scores))

    k = len(per_claim)
    variants = {"bm25_whole": whole}
    if k:
        variants["claim_max"] = [max(per_claim[c][d] for c in range(k)) for d in range(n_docs)]
        variants["claim_mean"] = [sum(per_claim[c][d] for c in range(k)) / k for d in range(n_docs)]
        variants["claim_min"] = [min(per_claim[c][d] for c in range(k)) for d in range(n_docs)]
        variants["claim_cover"] = [
            sum(1 for c in range(k) if per_claim[c][d] >= cover_tau) / k
            for d in range(n_docs)
        ]
        # coverage-first, similarity as tie-break
        variants["claim_cover_tb"] = [
            variants["claim_cover"][d] + 1e-3 * variants["claim_mean"][d]
            for d in range(n_docs)
        ]
        # hybrid: keep whole-query evidence, add coverage as a signal
        for lam in (0.3, 0.5):
            variants[f"hybrid_{lam}"] = [
                (1 - lam) * whole[d] + lam * variants["claim_mean"][d]
                for d in range(n_docs)
            ]
    return variants, per_claim
