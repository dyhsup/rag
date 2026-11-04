#!/usr/bin/env python3
"""下载 arXiv PDF 并用 pymupdf4llm 转成 Markdown。

用法: python3 scripts/download_parse.py selected.json
selected.json: [{"id": "2212.10496v1", "title": "...", "cites": 371}, ...]
"""
import json
import os
import sys
import time

import pymupdf4llm
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS = os.path.join(BASE, "papers")
MARKDOWN = os.path.join(BASE, "markdown")
PROXIES = {"http": os.environ.get("http_proxy", ""), "https": os.environ.get("https_proxy", "")} or None
HEADERS = {"User-Agent": "Mozilla/5.0 (research survey bot)"}


def safe_name(paper):
    base = paper["id"].split("v")[0].replace("/", "_")
    return base


def main():
    selected = json.load(open(sys.argv[1]))
    os.makedirs(PAPERS, exist_ok=True)
    os.makedirs(MARKDOWN, exist_ok=True)
    report = []
    for index, paper in enumerate(selected, 1):
        name = safe_name(paper)
        pdf_path = os.path.join(PAPERS, name + ".pdf")
        md_path = os.path.join(MARKDOWN, name + ".md")
        status = "exists"
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 20000:
            url = f"https://arxiv.org/pdf/{paper['id']}"
            try:
                response = requests.get(url, timeout=120, proxies=PROXIES, headers=HEADERS)
                if response.status_code != 200 or not response.content.startswith(b"%PDF"):
                    raise RuntimeError(f"HTTP {response.status_code} len={len(response.content)}")
                open(pdf_path, "wb").write(response.content)
                status = "downloaded"
            except Exception as exc:  # noqa: BLE001
                print(f"[{index}/{len(selected)}] {name} download ERR {repr(exc)[:120]}", flush=True)
                report.append({**paper, "pdf": None, "md": None, "error": repr(exc)[:200]})
                continue
            time.sleep(1.5)
        if not os.path.exists(md_path):
            try:
                md = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
                header = (
                    f"---\narxiv: {paper['id']}\ntitle: \"{paper.get('title', '')}\"\n"
                    f"year: {paper.get('year', '')}\ncites: {paper.get('cites', 0)}\n"
                    f"source: https://arxiv.org/abs/{paper['id']}\n---\n\n"
                )
                open(md_path, "w", encoding="utf-8").write(header + md)
                status += "+md"
            except Exception as exc:  # noqa: BLE001
                print(f"[{index}/{len(selected)}] {name} parse ERR {repr(exc)[:120]}", flush=True)
                report.append({**paper, "pdf": pdf_path, "md": None, "error": repr(exc)[:200]})
                continue
        size = os.path.getsize(pdf_path)
        md_size = os.path.getsize(md_path) if os.path.exists(md_path) else 0
        print(f"[{index}/{len(selected)}] {name} {status} pdf={size//1024}KB md={md_size//1024}KB | {paper.get('title','')[:70]}", flush=True)
        report.append({**paper, "pdf": pdf_path, "md": md_path, "pdf_bytes": size, "md_bytes": md_size})
    json.dump(report, open(os.path.join(BASE, "download_report.json"), "w"), ensure_ascii=False, indent=1)
    print("done", sum(1 for r in report if r.get("md")), "/", len(selected))


if __name__ == "__main__":
    main()
