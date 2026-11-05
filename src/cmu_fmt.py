"""Shared input formatting for the set-conditioned CMU scorer.

The cross-encoder sees (query + selected set) against (candidate). Selected
passages are clipped so the set stays inside the encoder window; the candidate
is clipped separately so truncation never eats it.
"""

Q_SEP = " [SEP] "
SEL_SEP = " || "
MAX_SEL_CHARS = 220
MAX_CAND_CHARS = 900


def text_a(q, sel):
    """Query plus the already-selected evidence set."""
    if not sel:
        return q
    parts = [s[:MAX_SEL_CHARS] for s in sel]
    return q + Q_SEP + SEL_SEP.join(parts)


def text_b(cand):
    return cand[:MAX_CAND_CHARS]
