#!/usr/bin/env python3
"""按 arXiv ID 校验并补入经典论文，避免相关性检索漏掉高引工作。"""
import json
import os
import xml.etree.ElementTree as ET

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXIES = {"http": os.environ.get("http_proxy", ""), "https": os.environ.get("https_proxy", "")} or None
NS = {"a": "http://www.w3.org/2005/Atom"}

CURATED = [
    "1904.08375",  # Document Expansion by Query Prediction (doc2query)
    "2212.10496",  # HyDE
    "2303.07678",  # Query2Doc
    "2305.14283",  # Rewrite-Retrieve-Read
    "2212.10509",  # IRCoT
    "2210.03350",  # Self-Ask
    "2205.10625",  # Least-to-Most Prompting
    "2310.11511",  # Self-RAG
    "2401.15884",  # Corrective RAG
    "2210.08726",  # RARR
    "2403.14403",  # Adaptive-RAG
    "2212.09477",  # ConvGQR
    "2305.03653",  # Query Expansion by Prompting LLMs
    "2305.07477",  # Generative Relevance Feedback
    "2410.13765",  # Knowledge-Aware Query Expansion
    "2411.13154",  # DMQR-RAG
    "2401.17043",  # LameR
    "2310.08559",  # Self-Knowledge guided Retrieval Augmentation
    "2404.00610",  # Query Rewriting survey?
    "2402.13178",  # Query rewriting in conversational search?
    "2506.00232",  # ComposeRAG
    "2504.20114",  # TreeHop
    "2405.14431",  # RaFe
    "2407.12529",  # Crafting the Path
]


def main():
    path = os.path.join(BASE, "candidates.json")
    cands = json.load(open(path))
    known = {c["id"].split("v")[0] for c in cands}
    ids = ",".join(CURATED)
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ids, "max_results": len(CURATED)},
        timeout=60,
        proxies=PROXIES,
    )
    root = ET.fromstring(response.content)
    added = 0
    for entry in root.findall("a:entry", NS):
        aid = entry.find("a:id", NS).text.rsplit("/", 1)[-1]
        base_id = aid.split("v")[0]
        if base_id in known:
            continue
        cands.append({
            "id": aid,
            "title": " ".join(entry.find("a:title", NS).text.split()),
            "abstract": " ".join(entry.find("a:summary", NS).text.split()),
            "published": entry.find("a:published", NS).text[:10],
            "authors": [a.find("a:name", NS).text for a in entry.findall("a:author", NS)],
            "queries": ["curated"],
        })
        added += 1
    json.dump(cands, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"curated added={added}, total={len(cands)}")


if __name__ == "__main__":
    main()
