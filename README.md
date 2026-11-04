# RAG Query–证据语义不匹配 调研

本目录是一次围绕 **"RAG 中 query 与检索证据语义不匹配"** 的文献调研产物：23 篇 arXiv 论文的 PDF、Markdown 解析文本、元数据，以及两份中文综述文档。

## 先读什么

| 文件 | 内容 |
|---|---|
| [`HANDOVER.md`](HANDOVER.md) | **交接文档**：环境约束、已完成/未完成、研究方案（SALR）、实验协议、从零搭建步骤、Prompt/Schema、已知坑 |
| [`survey.md`](survey.md) | 主综述：问题界定（四类失配）、方法分类、对比表、实验证据、规律、评测、开放问题 |
| [`top10.md`](top10.md) | 引用最高的 10 篇论文逐篇精读（方法、实验数字、局限、工程启示） |
| [`papers.csv`](papers.csv) | 23 篇论文的元数据：arXiv ID、标题、年份、OpenAlex 引用数、文件路径 |
| `papers/` | 原始 PDF（28 MB） |
| `markdown/` | PyMuPDF4LLM 解析的 Markdown（含标题、表格、公式文本） |
| `scripts/` | 可复现脚本：`enrich_cites.py`（引用数校准）、`add_curated.py`（补充经典论文）、`download_parse.py`（下载+解析） |

## 一句话结论

query–证据不匹配有四层（词面、粒度、表示、证据有效性），对应四类方法：**生成式扩展（HyDE/Query2Doc）**、**改写与分解（Rewrite-Retrieve-Read/IRCoT）**、**索引侧扩展（doc2query）**、**检索后纠错与归因（CRAG/Self-RAG/RARR）**，外加**复杂度路由（Adaptive-RAG）**做性价比控制。

## Top 10（按 OpenAlex 引用数，2026-11-04）

1. Least-to-Most Prompting（388）
2. HyDE（371）
3. IRCoT（359）
4. Self-Ask / Compositionality Gap（303）
5. Rewrite-Retrieve-Read（276）
6. Adaptive-RAG（264）
7. Query2doc（220）
8. doc2query（215）
9. CRAG（187）
10. RARR（148）

## 复现方式

```bash
export http_proxy="http://<user>:<pass>@<host>:<port>"
export https_proxy="$http_proxy"

python3 scripts/enrich_cites.py       # OpenAlex 按标题校准引用数（可断点续跑）
python3 scripts/add_curated.py        # 补入经典论文（按 arXiv ID 校验）
python3 scripts/download_parse.py selected.json   # 下载 PDF + 转 Markdown
```

## 说明与局限

- **引用数**来自 OpenAlex，普遍低于 Google Scholar（例如 HyDE 在 Google Scholar 上远高于 371），仅用于排序，不代表绝对影响力。
- 检索词覆盖"hypothetical document / query rewriting / query expansion / decomposition / vocabulary mismatch / corrective RAG / self-reflection"等 15 组 arXiv 查询，共 288 个候选，再人工筛选到 23 篇；**不保证穷尽**，尤其是 2026 年下半年的新工作。
- 各论文的数据集、底座模型不同，跨论文数字不可直接横向比较。
- Markdown 由 PDF 自动解析，表格与公式可能有排版噪声；引用具体数字时请回看 PDF。
