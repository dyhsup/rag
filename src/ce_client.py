"""Client for the remote cross-encoder / bi-encoder service.

Scores returned by /rerank and /score are RAW LOGITS; use sigmoid() before
thresholding. Internal endpoint, so proxies are always bypassed.
"""
import os
import time

import numpy as np
import requests


class CERemote:
    def __init__(self, url="http://10.52.100.210:8833", timeout=300, retries=3, chunk=128):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.chunk = chunk
        self.dataset_url = os.environ.get("CE_DATASET_URL")
        self.dataset = os.environ.get("CE_DATASET", "musique")
        if self.dataset_url:
            self.url = self.dataset_url.rstrip("/")
        self.embed_url = (os.environ.get("CE_EMBED_URL") or self.url).rstrip("/")
        self.local_path = os.environ.get("CE_LOCAL_PATH")
        if self.local_path:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._torch = torch
            torch.set_num_threads(int(os.environ.get("CE_THREADS", "8")))
            self._tok = AutoTokenizer.from_pretrained(self.local_path)
            self._mdl = AutoModelForSequenceClassification.from_pretrained(self.local_path).eval()

    def _local_scores(self, pairs, max_len=256):
        out = []
        with self._torch.inference_mode():
            for i in range(0, len(pairs), self.chunk):
                c = pairs[i:i + self.chunk]
                b = self._tok([x[0] for x in c], [x[1] for x in c], truncation=True,
                              max_length=max_len, padding=True, return_tensors="pt")
                out.extend(self._mdl(**b).logits.view(-1).tolist())
        return out

    def _post(self, path, payload):
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.url}{path}", json=payload,
                                  timeout=self.timeout, proxies={})
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last = repr(e)
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"{path} failed: {last}")

    def health(self):
        return requests.get(f"{self.url}/health", timeout=10, proxies={}).json()

    def rerank(self, query, documents):
        """Logits aligned to the input order."""
        if self.dataset_url:
            out = [0.0] * len(documents)
            for i in range(0, len(documents), self.chunk):
                part = documents[i:i + self.chunk]
                data = self._post(f"/rerank/{self.dataset}", {"query": query, "documents": part})
                for r in data["results"]:
                    out[i + r["index"]] = r["score"]
            return out
        if self.local_path:
            return self._local_scores([[query, d] for d in documents])
        out = [0.0] * len(documents)
        for i in range(0, len(documents), self.chunk):
            part = documents[i:i + self.chunk]
            data = self._post("/rerank", {"query": query, "documents": part})
            for r in data["results"]:
                out[i + r["index"]] = r["score"]
        return out

    def score(self, pairs):
        if self.dataset_url:
            out = []
            for i in range(0, len(pairs), self.chunk):
                data = self._post(f"/score/{self.dataset}", {"pairs": [list(p) for p in pairs[i:i + self.chunk]]})
                out.extend(data["scores"])
            return out
        if self.local_path:
            return self._local_scores([list(p) for p in pairs])
        out = []
        for i in range(0, len(pairs), self.chunk):
            data = self._post("/score", {"pairs": [list(p) for p in pairs[i:i + self.chunk]]})
            out.extend(data["scores"])
        return out

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), self.chunk):
            r = requests.post(f"{self.embed_url}/embed",
                              json={"texts": texts[i:i + self.chunk], "normalize": True},
                              timeout=self.timeout, proxies={})
            if r.status_code != 200:
                raise RuntimeError(f"/embed HTTP {r.status_code}: {r.text[:120]}")
            data = r.json()
            out.extend(data["embeddings"])
        return np.asarray(out, dtype=np.float32)
