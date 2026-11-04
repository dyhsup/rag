"""OpenAI-compatible LLM client: endpoint rotation, sqlite cache, JSON parsing.

Internal endpoints must bypass the proxy (see HANDOVER 1.3).
"""
import hashlib
import json
import re
import sqlite3
import threading
import time

import requests

ENDPOINTS = [
    "http://10.94.199.216:8000/v1",
    "http://10.94.199.217:8000/v1",
]
MODEL = "qwen3.6-27b"


class LLM:
    def __init__(self, cache_path="cache/llm.sqlite"):
        self.cache_path = cache_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._rr = 0
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0

    # -- cache -------------------------------------------------------------
    def _conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.cache_path, timeout=30)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)"
            )
            self._local.conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INTEGER)"
            )
            self._local.conn.commit()
        return self._local.conn

    def _get(self, key):
        row = self._conn().execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def _put(self, key, value):
        with self._lock:
            c = self._conn()
            c.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, value))
            c.commit()

    # -- call --------------------------------------------------------------
    def chat(self, messages, max_tokens=512, temperature=0.0, retries=3):
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        key = hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        hit = self._get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit

        last_err = None
        for attempt in range(retries):
            with self._lock:
                ep = ENDPOINTS[self._rr % len(ENDPOINTS)]
                self._rr += 1
            try:
                r = requests.post(
                    f"{ep}/chat/completions",
                    json=payload,
                    timeout=120,
                    proxies={},  # internal endpoints: never proxy
                )
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                out = r.json()["choices"][0]["message"]["content"]
                self.calls += 1
                self._put(key, out)
                return out
            except Exception as e:  # endpoint down, timeout, bad json
                last_err = e
                self.failures += 1
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

    def json_call(self, prompt, max_tokens=768, expect="object", retries=2):
        """Call and parse JSON; on parse failure, ask the model to repair once."""
        last = None
        for _ in range(retries):
            text = self.chat(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens
            )
            obj = extract_json(text, expect=expect)
            if obj is not None:
                return obj
            last = text
            prompt = (
                prompt
                + "\n\nYour previous output was not valid JSON. Output ONLY the JSON, "
                "no markdown fence, no explanation."
            )
        raise ValueError(f"unparseable JSON: {str(last)[:300]}")


def extract_json(text, expect="object"):
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    candidates = []
    if expect == "object":
        candidates = [(t.find("{"), t.rfind("}") + 1)]
    else:
        candidates = [(t.find("["), t.rfind("]") + 1), (t.find("{"), t.rfind("}") + 1)]
    for a, b in candidates:
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b])
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None
