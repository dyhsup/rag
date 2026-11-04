#!/usr/bin/env python3
"""用 OpenAlex 按标题匹配，校正候选论文的引用数。可断点续跑。"""
import difflib
import json
import os
import time
import urllib.parse

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXIES = {"http": os.environ.get("http_proxy", ""), "https": os.environ.get("https_proxy", "")} or None


def norm(text):
    return " ".join((text or "").lower().split())


def main():
    path = os.path.join(BASE, "candidates.json")
    cands = json.load(open(path))
    for index, cand in enumerate(cands):
        if cand.get("oa_cites") is not None:
            continue
        url = (
            "https://api.openalex.org/works?search="
            + urllib.parse.quote(cand["title"])
            + "&select=title,cited_by_count,publication_year&per-page=5&mailto=research@example.com"
        )
        best = 0
        best_title = ""
        try:
            response = requests.get(url, timeout=30, proxies=PROXIES)
            for work in response.json().get("results", []):
                sim = difflib.SequenceMatcher(None, norm(cand["title"]), norm(work.get("title"))).ratio()
                if sim > 0.85 and work.get("cited_by_count", 0) > best:
                    best = work["cited_by_count"]
                    best_title = work.get("title") or ""
        except Exception as exc:  # noqa: BLE001
            print("ERR", cand["id"], repr(exc)[:100], flush=True)
            cand["oa_cites"] = 0
            continue
        cand["oa_cites"] = best
        cand["oa_title"] = best_title
        if index % 25 == 0:
            print(f"{index}/{len(cands)} {cand['id']} cites={best}", flush=True)
            json.dump(cands, open(path, "w"), ensure_ascii=False, indent=1)
        time.sleep(0.12)
    json.dump(cands, open(path, "w"), ensure_ascii=False, indent=1)
    matched = sum(1 for c in cands if c.get("oa_cites"))
    print(f"done matched={matched}/{len(cands)}", flush=True)


if __name__ == "__main__":
    main()
