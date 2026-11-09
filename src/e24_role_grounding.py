"""E24: requirement-structured first-stage grounding.

Role-OT is used only as an implementation of a structured grounding score.
The research object is the requirement/evidence alignment over entity roles,
relation direction, and answer type -- not a new OT retriever.

This first prototype is training-free with respect to matching, but its
screening protocol uses MuSiQue gold plans and gold previous-hop answers to
render requirements. It therefore estimates the oracle-binding condition and
must not be interpreted as an end-to-end predicted-binding result. The
structural score itself does not read the final answer or gold passage labels.
It provides a deterministic structural score, a CE+structure hybrid, and the
existing requirement-CE baseline on the same 20-passage pool.
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from .ce_client import CERemote
from .e3_binding import em, f1, render
from .llm import LLM
from .musique import load
from .e4_pipeline import READ, fmt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

STOP = set("""a an the what who whom which where when in on of is was were did do does
are to for from by with and or as at into current administrative territorial
entity year university person film book author located went get got play plays
played old new today spouse director directed author written wrote actor actress
role character city country state province university college educated school
attended year date born founded built opened occupied nationality national citizen
award won winner palme prize""".split())
REL_GROUPS = {
    "spouse": {"spouse", "married", "marriage", "husband", "wife"},
    "director": {"director", "directed", "directed_by"},
    "author": {"author", "written", "wrote", "book"},
    "actor": {"actor", "actress", "played", "plays", "role", "character"},
    "location": {"located", "location", "city", "country", "state", "province"},
    "education": {"university", "college", "educated", "school", "attended", "went"},
    "date": {"year", "date", "born", "founded", "built", "opened", "occupied"},
    "nationality": {"nationality", "national", "citizen"},
    "award": {"award", "won", "winner", "palme", "prize"},
}
TYPE_WORDS = {
    "person": {"who", "person", "actor", "actress", "director", "author", "spouse", "husband", "wife"},
    "location": {"where", "city", "country", "state", "province", "location", "university"},
    "date": {"year", "when", "date", "born", "founded", "built", "opened"},
    "title": {"film", "book", "title", "movie"},
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOKEN_RE.findall(s.lower()))


def relation_features(s):
    ts = toks(s)
    active = set()
    for name, words in REL_GROUPS.items():
        if ts & words:
            active.add(name)
    return active


def type_features(s):
    ts = toks(s)
    return {name for name, words in TYPE_WORDS.items() if ts & words}


def entity_tokens(req):
    """Approximate head/entity tokens without using the gold answer."""
    ts = toks(req)
    return {x for x in ts if x not in STOP and not x.isdigit() and len(x) > 2}


def sentence_parts(text):
    return [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x] or [text]


def structural_cost(req, passage):
    """A lightweight OT-style cost over requirement/evidence attributes.

    We create one requirement token mass and sentence/token evidence masses.
    The cost rewards entity overlap, relation-family overlap, type compatibility,
    and direction markers. It does not inspect q['answer'] or gold labels.
    """
    rt, et = toks(req), toks(passage)
    ents = entity_tokens(req)
    ent_overlap = len(ents & et) / max(1, len(ents))
    rel = relation_features(req)
    ev_rel = relation_features(passage)
    rel_overlap = len(rel & ev_rel) / max(1, len(rel)) if rel else 0.0
    typ = type_features(req)
    ev_typ = type_features(passage)
    type_overlap = 1.0 if typ & ev_typ else 0.0
    # Local sentence support approximates transport to the best evidence span.
    local = 0.0
    for sent in sentence_parts(passage):
        st = toks(sent)
        e = len(ents & st) / max(1, len(ents))
        r = len(rel & relation_features(sent)) / max(1, len(rel)) if rel else 0.0
        local = max(local, 0.65 * e + 0.25 * r + 0.10 * type_overlap)
    # Penalise a pure topical match with no requested relation signal.
    topical = len((rt - STOP) & et) / max(1, len(rt - STOP))
    score = 0.45 * ent_overlap + 0.25 * local + 0.20 * rel_overlap + 0.10 * type_overlap
    if topical > 0.35 and rel and rel_overlap == 0:
        score -= 0.08
    return max(0.0, min(1.0, score))


def minmax(xs):
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def select_seq(per_req, k):
    selected = []
    for scores in per_req:
        if len(selected) >= k:
            break
        rest = [i for i in range(len(scores)) if i not in selected]
        if rest:
            selected.append(max(rest, key=lambda i: scores[i]))
    return selected


def select_cover(per_req, k):
    norm = [minmax(x) for x in per_req]
    cur = [0.0] * len(norm)
    selected = []
    while len(selected) < k:
        rest = [i for i in range(len(norm[0])) if i not in selected]
        if not rest:
            break
        i = max(rest, key=lambda j: sum(max(0.0, norm[r][j] - cur[r]) for r in range(len(norm))))
        selected.append(i)
        for r in range(len(norm)):
            cur[r] = max(cur[r], norm[r][i])
    return selected


def evaluate(n=100, k=2, model=None, threads=4, read=False, tag="e24_role_ot"):
    qs = load()[:n]
    ce = CERemote() if model is None else CERemote()
    # Force local CE when a path is supplied; CERemote reads CE_LOCAL_PATH.
    if model:
        os.environ["CE_LOCAL_PATH"] = model
        ce = CERemote()
    llm = LLM(os.path.join(ROOT, "cache", "llm.sqlite")) if read else None

    def one(q):
        docs = [d["text"] for d in q["docs"]][:20]
        gold = {i for i, d in enumerate(q["docs"][:20]) if d["is_gold"]}
        reqs = [render(h["question"], [x["answer"] for x in q["hops"][:h["hop"]]]) or q["question"] for h in q["hops"][:4]]
        ce_per = [ce.rerank(r, docs) for r in reqs]
        struct_docs = [str(q["docs"][i].get("title", "")) + " . " + docs[i] for i in range(len(docs))]
        struct_per = [[structural_cost(r, d) for d in struct_docs] for r in reqs]
        hybrid_per = [[0.7 * c + 0.3 * s for c, s in zip(ce_row, st_row)]
                      for ce_row, st_row in zip(ce_per, struct_per)]
        methods = {
            "seq_ce": select_seq(ce_per, k),
            "role_ot": select_seq(struct_per, k),
            "role_ot_cover": select_cover(struct_per, k),
            "hybrid": select_seq(hybrid_per, k),
        }
        rec = {"qid": q["qid"], "n_req": len(reqs), "n_gold": len(gold), "reqs": reqs}
        for name, sel in methods.items():
            rec[name + "_sel"] = sel
            rec[name + "_cov"] = len(gold & set(sel)) / max(1, len(gold))
            rec[name + "_full"] = int(gold.issubset(set(sel)))
            if read:
                try:
                    pred = llm.chat([{"role": "user", "content": READ.format(q=q["question"], ps=fmt([docs[i] for i in sel]))}], max_tokens=32).strip().split("\n")[0]
                except Exception:
                    pred = ""
                rec[name + "_em"] = em(pred, q["answer"])
                rec[name + "_f1"] = f1(pred, q["answer"])
        return rec

    with ThreadPoolExecutor(max_workers=threads) as ex:
        recs = list(ex.map(one, qs))
    summary = {"n": len(recs), "k": k, "plan": "gold", "binding": "gold", "candidate_pool": "per_question_20", "model": model or "remote"}
    for name in ("seq_ce", "role_ot", "role_ot_cover", "hybrid"):
        for metric in ("cov", "full") + (("em", "f1") if read else ()):
            summary[name + "_" + metric] = sum(r[name + "_" + metric] for r in recs) / len(recs)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, tag + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": recs}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--model", default="dist/reqce/model_hop")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--tag", default="e24_role_ot")
    a = ap.parse_args()
    evaluate(a.n, a.k, a.model, a.threads, a.read, a.tag)
