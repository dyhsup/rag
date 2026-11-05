"""RBG with hybrid retrieval (BM25 union dense) for multi-hop."""
import argparse
import os

import numpy as np

from . import e13_exec
from .ce_client import CERemote

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ORIG_TOPK = e13_exec.topk
_cache = {"ce": None, "emb": {}}


def hybrid_topk(index, text, k=50):
    if _cache["ce"] is None:
        _cache["ce"] = CERemote()
    ce = _cache["ce"]
    key = (id(index), len(index.docs))
    if key not in _cache["emb"]:
        _cache["emb"][key] = ce.embed([d["text"] for d in index.docs])
    de = _cache["emb"][key]
    cand_bm, _ = _ORIG_TOPK(index, text, k)
    try:
        qe = ce.embed([text])[0]
        order = list(np.argsort(-(de @ qe)))[:k]
        cand_dn = [index.docs[i]["text"] for i in order]
    except Exception:
        cand_dn = []
    return list(dict.fromkeys(cand_bm + cand_dn))[:2 * k], None


def main(n=200, workers=10, data="wiki2_200.jsonl"):
    e13_exec.topk = hybrid_topk
    e13_exec.run(n, workers, ("select",), data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="wiki2_200.jsonl")
    a = ap.parse_args()
    main(a.n, 10, a.data)
