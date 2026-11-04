"""A-line: project a question into atomic claims, then score passages by
coverage (sum / min / max) instead of a single whole-query similarity."""
import re

CLAIM_PROMPT = """You are an information-retrieval planner. Rewrite the question as the minimal set of atomic claims that the evidence must support. Output JSON only.

Rules:
1. Each claim is a declarative sentence describing what the evidence would say, using "?" for the unknown value.
2. Every entity must appear verbatim in the question. Never add facts from your own knowledge.
3. If the question compares two entities (A and B ...), emit ONE CLAIM PER ENTITY.
4. If the question chains facts (the X of Y that Z), emit ONE CLAIM PER HOP. Never merge two hops into one claim.
5. Keep the entity that the question binds to this hop inside the claim text. Only write "$c1" as the subject when the subject truly is the unknown answer of an earlier claim; never write a claim whose only subject is "$c1".
6. The final claim must contain the placeholder "?".
7. Use 1-4 claims. relation is snake_case.
8. answer_type in {person, date, number, location, organization, title, boolean, other}.

Question: Who directed the film that won the Palme d'Or in 2019?
Output:
{"answer_type": "person", "claims": [{"id": "c1", "text": "a film won the Palme d'Or in 2019", "head": "?", "relation": "award_year_to_film", "object": "Palme d'Or (2019)", "type": "title"}, {"id": "c2", "text": "$c1 was directed by ?", "head": "$c1", "relation": "directed_by", "object": "?", "type": "person"}]}

Question: What nationality was Oliver Reed's character in the film Royal Flash?
Output:
{"answer_type": "other", "claims": [{"id": "c1", "text": "Oliver Reed played ? in the film Royal Flash", "head": "Oliver Reed", "relation": "actor_role_in_film", "object": "Royal Flash", "type": "person"}, {"id": "c2", "text": "$c1 had nationality ?", "head": "$c1", "relation": "nationality", "object": "?", "type": "other"}]}

Question: Are both Scott Derrickson and Ed Wood of the same nationality?
Output:
{"answer_type": "boolean", "claims": [{"id": "c1", "text": "Scott Derrickson is of nationality ?", "head": "Scott Derrickson", "relation": "nationality", "object": "?", "type": "boolean"}, {"id": "c2", "text": "Ed Wood is of nationality ?", "head": "Ed Wood", "relation": "nationality", "object": "?", "type": "boolean"}]}

Question: {question}
Output:
"""

_REL_RE = re.compile(r"^[a-z0-9_]+$")


def _bad_plan(obj):
    """A plan is bad if it is empty, has no answer placeholder, or a claim
    references itself / a later claim ($cX with X >= its own position)."""
    claims = obj.get("claims", [])
    if not claims:
        return True
    for i, c in enumerate(claims, 1):
        for ref in re.findall(r"\$c(\d+)", c.get("text") or ""):
            if int(ref) >= i:
                return True
    return not any("?" in (c.get("text") or "") for c in claims)


def project(llm, question):
    prompt = CLAIM_PROMPT.replace("{question}", question)
    obj = llm.json_call(prompt, max_tokens=512)
    if _bad_plan(obj):
        # one targeted repair attempt, then accept whatever we have
        obj2 = llm.json_call(
            prompt
            + "\n\nCorrection: claims must be ordered so that $cX only refers to an"
            " earlier claim, and the last claim must contain the placeholder '?'."
            " Output ONLY the JSON.",
            max_tokens=512,
        )
        if not _bad_plan(obj2):
            obj = obj2
    claims = []
    for i, c in enumerate(obj.get("claims", [])[:4]):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        rel = (c.get("relation") or "related_to").strip()
        if not _REL_RE.match(rel):
            rel = "related_to"
        claims.append(
            {
                "id": f"c{len(claims)+1}",
                "text": text,
                "head": (c.get("head") or "").strip(),
                "relation": rel,
                "object": (c.get("object") or "").strip(),
                "type": (c.get("type") or "other").strip(),
            }
        )
    if not claims:  # fall back to the raw question, so the pipeline never dies
        claims = [
            {
                "id": "c1",
                "text": question.rstrip("?") + " ?",
                "head": "",
                "relation": "related_to",
                "object": "",
                "type": obj.get("answer_type", "other"),
            }
        ]
    return {"answer_type": obj.get("answer_type", "other"), "claims": claims}


def claim_query_text(claim):
    """Text used for matching: placeholders and slot refs removed."""
    t = claim["text"].replace("$c", " ").replace("?", " ")
    return re.sub(r"\s+", " ", t).strip()
