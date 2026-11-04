"""B-line (oracle upper bound): one LLM call per (question, passage) that
judges slot coverage directly and returns verbatim spans.

If even this cannot beat BM25, the whole alignment premise fails.
"""
import re

JUDGE_PROMPT = """You judge whether a passage contains the facts a question needs. Use ONLY the passage; no outside knowledge.

Question: {question}
Information needs:
{needs}

Passage (title: {title}):
{text}

For each information need, decide whether the passage explicitly states that fact (or an unambiguous paraphrase of it).
Output JSON only:
{{"coverage": [{{"id": "c1", "covered": true, "span": "verbatim text copied from the passage"}}]}}

Rules:
- covered = true only when the passage itself supports the need.
- span must be copied verbatim from the passage; if not supported, use "".
- Include one entry per information need, in order.
"""


def _norm_ws(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def judge(llm, question, claims, doc):
    needs = "\n".join(f"{c['id']}: {c['text']}" for c in claims)
    prompt = JUDGE_PROMPT.format(
        question=question,
        needs=needs,
        title=doc["title"],
        text=doc["text"][:4000],
    )
    obj = llm.json_call(prompt, max_tokens=768)
    got = {x.get("id"): x for x in obj.get("coverage", []) if isinstance(x, dict)}

    flags, span_ok, span_total = [], 0, 0
    for c in claims:
        x = got.get(c["id"], {})
        flags.append(bool(x.get("covered")))
        span = (x.get("span") or "").strip()
        if span:
            span_total += 1
            if _norm_ws(span) in _norm_ws(doc["text"]):
                span_ok += 1

    return {
        "covered": flags,
        "covered_count": sum(flags),
        "score": sum(flags) / max(1, len(claims)),
        "span_total": span_total,
        "span_valid": span_ok,
    }
