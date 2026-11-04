"""Client for the remote cross-encoder / bi-encoder service.

Scores returned by /rerank and /score are RAW LOGITS; use sigmoid() before
thresholding. Internal endpoint, so proxies are always bypassed.
"""
import time

import numpy as np
import requests


class CERemote:
    def __init__(self, url="http://10.52.100.210:8833", timeout=300, retries=3, chunk=128):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.chunk = chunk

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
        out = [0.0] * len(documents)
        for i in range(0, len(documents), self.chunk):
            part = documents[i:i + self.chunk]
            data = self._post("/rerank", {"query": query, "documents": part})
            for r in data["results"]:
                out[i + r["index"]] = r["score"]
        return out

    def score(self, pairs):
        out = []
        for i in range(0, len(pairs), self.chunk):
            data = self._post("/score", {"pairs": [list(p) for p in pairs[i:i + self.chunk]]})
            out.extend(data["scores"])
        return out

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), self.chunk):
            data = self._post("/embed", {"texts": texts[i:i + self.chunk], "normalize": True})
            out.extend(data["embeddings"])
        return np.asarray(out, dtype=np.float32)
