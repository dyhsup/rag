"""Minimal cross-encoder / bi-encoder HTTP service.

Run:
    pip install -U transformers torch fastapi uvicorn
    CE_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    TORCH_THREADS=8 uvicorn ce_server:app --host 0.0.0.0 --port 8080 --workers 1

Endpoints
    GET  /health                      -> {"status":"ok","models":{...}}
    POST /rerank  {query, texts, batch_size?, max_length?, raw_scores?}
                                      -> {"scores":[float...], "model":str}
    POST /score   {pairs:[[q,p],...]} -> {"scores":[float...], "model":str}
    POST /embed   {texts:[...]}       -> {"embeddings":[[float...],...], "model":str}

Scores are sigmoid(logit) in [0,1] unless raw_scores=true.
"""
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

CE_MODEL = os.environ.get("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
THREADS = int(os.environ.get("TORCH_THREADS", "8"))
torch.set_num_threads(THREADS)

app = FastAPI()

ce_tok = AutoTokenizer.from_pretrained(CE_MODEL)
ce_mdl = AutoModelForSequenceClassification.from_pretrained(CE_MODEL).eval()

emb_tok = None
emb_mdl = None


class RerankReq(BaseModel):
    query: str
    texts: list[str]
    batch_size: int = 64
    max_length: int = 256
    raw_scores: bool = False


class ScoreReq(BaseModel):
    pairs: list[list[str]]
    batch_size: int = 64
    max_length: int = 256
    raw_scores: bool = False


class EmbedReq(BaseModel):
    texts: list[str]
    batch_size: int = 64
    max_length: int = 256


def _ce_scores(pairs, batch_size, max_length, raw):
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            b = ce_tok([p[0] for p in chunk], [p[1] for p in chunk],
                       padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt")
            logits = ce_mdl(**b).logits.view(-1)
            out.extend((logits if raw else torch.sigmoid(logits)).tolist())
    return out


@app.get("/health")
def health():
    return {"status": "ok", "models": {"ce": CE_MODEL, "embed": EMB_MODEL,
                                        "threads": torch.get_num_threads()}}


@app.post("/rerank")
def rerank(req: RerankReq):
    scores = _ce_scores([[req.query, t] for t in req.texts],
                        req.batch_size, req.max_length, req.raw_scores)
    return {"scores": scores, "model": CE_MODEL}


@app.post("/score")
def score(req: ScoreReq):
    scores = _ce_scores(req.pairs, req.batch_size, req.max_length, req.raw_scores)
    return {"scores": scores, "model": CE_MODEL}


@app.post("/embed")
def embed(req: EmbedReq):
    global emb_tok, emb_mdl
    if emb_mdl is None:
        emb_tok = AutoTokenizer.from_pretrained(EMB_MODEL)
        emb_mdl = AutoModel.from_pretrained(EMB_MODEL).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(req.texts), req.batch_size):
            b = emb_tok(req.texts[i:i + req.batch_size], padding=True, truncation=True,
                        max_length=req.max_length, return_tensors="pt")
            h = emb_mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            e = torch.nn.functional.normalize(e, dim=-1)
            out.extend(e.tolist())
    return {"embeddings": out, "model": EMB_MODEL}
