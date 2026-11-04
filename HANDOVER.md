# 交接文档：RAG Query–证据语义对齐调研与 SALR 方案

> 生成时间：2026-11-04
> 工作目录：`/disk1/rag/query-evidance`（本文档所有路径均相对于该目录）
> 状态：**调研已完成；研究方案已设计；实验尚未执行**
> 说明：本文档自包含，不依赖任何上级目录或外部实验目录。环境隔离后，按第 7 节可从零重建。

---

## 0. 一页纸摘要

> **进展更新（2026-11-04，本次会话）**：20 题 pilot 已跑完，代码在 `src/`，结论与下一步见 [`PILOT.md`](PILOT.md)。核心结果：LLM 槽位覆盖判定（oracle 上界）MRR +0.147 [+0.001,+0.304]；BM25 打分的廉价近似只回收约 1/3 增益，瓶颈在匹配函数而非分解；覆盖信号应叠加在 BM25 上而非替换。

### 已完成
- 检索并筛选 **23 篇**与 "RAG 中 query 与证据语义不匹配" 直接相关的论文；
- 下载 PDF（`papers/`，28 MB），用 PyMuPDF4LLM 解析为 Markdown（`markdown/`，1.8 MB）；
- 用 OpenAlex 校准引用数，形成排序；
- 产出中文综述 `survey.md`、Top10 精读 `top10.md`、索引 `README.md`、元数据 `papers.csv`。

### 未完成
- **没有跑任何实验**。所有实验设计只存在于文档中。
- 研究方案尚未得到最终确认（用户最后一次指示：动机要更直接地解决 query–证据语义对齐，因此方案从"失败后修复"改为"检索时对齐"，即 SALR）。

### 下一步（唯一优先级）
在隔离环境重建数据与检索索引，跑 **20 题 pilot**，验证三件事：
1. 槽位解析成功率；
2. 对齐分数能否把金标证据排进 top-5（对比 BM25）；
3. 关系同义词子集（head vs director）上是否有可见改善。

---

## 1. 环境与约束

### 1.1 硬件（实测）
| 项 | 值 |
|---|---|
| GPU | **无**（`nvidia-smi` 不存在，`torch.cuda.is_available() == False`） |
| CPU | 8 核 |
| 内存 | 31 GB（检查时仅约 3 GB 空闲，注意 BM25 索引加载会吃内存） |
| 磁盘 | `/disk1` 剩余约 181 GB |

### 1.2 软件
| 包 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.4.0+cu121（CUDA 不可用） |
| transformers | 4.44.2 |
| datasets | 3.0.0 |
| faiss | 可用 |
| pymupdf / pymupdf4llm | 1.24.14 / 0.0.17 |
| pdftotext | 可用（备用解析器） |
| requests | 可用 |

> `pyserini` 未安装；BM25 建议用纯 Python 的 `rank_bm25`，避免 Java 依赖。

### 1.3 网络
- **外网需代理**：`http://<PROXY_USER>:<PROXY_PASS>@10.251.112.50:8128`
  - 凭据由负责人提供，**不要写入代码或提交到仓库**；建议用环境变量 `http_proxy` / `https_proxy`。
- **内网 API 无需代理**，且必须显式绕过代理（`--noproxy '*'` 或 `proxies={}`）。
- `curl` 7.19.7（NSS）对 arXiv 等站点 TLS 握手失败，**统一改用 Python `requests`**。

### 1.4 可用的 LLM 服务（vLLM，OpenAI 兼容）
| 端点 | 状态 | 模型 |
|---|---|---|
| `http://10.94.199.217:8000/v1` | 可用 | `qwen3.6-27b` |
| `http://10.94.199.216:8000/v1` | 可用 | `qwen3.6-27b` |
| `http://10.94.199.218:8000/v1` | **不可达** | — |

**必须关闭思考模式**，否则输出会包含 "Here's a thinking process:"：

```python
payload = {
    "model": "qwen3.6-27b",
    "messages": [...],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
}
```

实测：关闭后 0.13 s 返回 3 个 token；开启时 0.9 s 且内容被思考过程占满。

---

## 2. 已交付物清单

```
/disk1/rag/query-evidance/
├── HANDOVER.md            # 本文档
├── README.md              # 目录索引（面向读者）
├── survey.md              # 主综述（问题界定/分类/对比/实验证据/开放问题）
├── top10.md               # 引用最高 10 篇的逐篇精读
├── papers.csv             # 23 篇元数据：arXiv ID、标题、年份、引用、路径
├── selected.json          # 入选 23 篇（含摘要）
├── candidates.json        # 300 个候选（含引用校准结果，见第 4 节缺陷）
├── download_report.json   # 下载/解析结果与字节数
├── papers/                # 23 个 PDF
├── markdown/              # 23 个 Markdown（YAML frontmatter + 正文）
└── scripts/
    ├── enrich_cites.py    # OpenAlex 按标题校准引用数（断点续跑）
    ├── add_curated.py     # 按 arXiv ID 补入经典论文
    └── download_parse.py  # 下载 PDF + pymupdf4llm 转 Markdown
```

### 2.1 23 篇论文（按引用数降序）

| # | arXiv | 简称 | 年份 | 引用 | 类别 |
|---|---|---|---|---|---|
| 1 | 2205.10625 | Least-to-Most | 2022 | 388 | 分解 |
| 2 | 2212.10496 | HyDE | 2022 | 371 | 假设文档 |
| 3 | 2212.10509 | IRCoT | 2022 | 359 | 迭代检索 |
| 4 | 2210.03350 | Self-Ask | 2022 | 303 | 分解 |
| 5 | 2305.14283 | Rewrite-Retrieve-Read | 2023 | 276 | 改写 |
| 6 | 2403.14403 | Adaptive-RAG | 2024 | 264 | 控制层 |
| 7 | 2303.07678 | Query2Doc | 2023 | 220 | 假设文档 |
| 8 | 1904.08375 | doc2query | 2019 | 215 | 索引侧 |
| 9 | 2401.15884 | CRAG | 2024 | 187 | 检索后 |
| 10 | 2210.08726 | RARR | 2022 | 148 | 检索后 |
| 11 | 2103.04831 | First-stage 综述 | 2021 | 119 | 表示层 |
| 12 | 2310.11511 | Self-RAG | 2023 | 104 | 生成侧反思 |
| 13 | 2304.13157 | GRF | 2023 | 49 | 生成式反馈 |
| 14 | 2305.03653 | QE by Prompting | 2023 | 26 | 扩展 |
| 15 | 2405.14431 | RaFe | 2024 | 21 | 改写（RL） |
| 16 | 2310.19056 | MILL | 2023 | 17 | 扩展（互验） |
| 17 | 2401.06311 | QE Best Practices | 2024 | 15 | 扩展 |
| 18 | 2410.13765 | Knowledge-Aware QE | 2024 | 5 | 扩展（关系） |
| 19 | 2407.12529 | Crafting the Path | 2024 | 4 | 改写 |
| 20 | 2607.29402 | HyPE | 2026 | 4 | 假设提示嵌入 |
| 21 | 2412.17558 | Query Optimization 综述 | 2024 | 2 | 综述 |
| 22 | 2411.13154 | DMQR-RAG | 2024 | 2 | 多查询改写 |
| 23 | 2504.20114 | TreeHop | 2025 | 0 | embedding 层改写 |

> 引用数为 OpenAlex（2026-11-04），普遍低于 Google Scholar，仅用于相对排序。

---

## 3. 调研阶段：方法与可复现步骤

### 3.1 检索策略
用 15 组 arXiv API 查询召回 288 个候选，覆盖：
`hypothetical document embeddings` / `query-document mismatch` / `semantic gap + retrieval-augmented` / `query rewriting + RAG` / `query expansion + LLM` / `query decomposition + retrieval` / `vocabulary mismatch` / `query + evidence + mismatch + RAG` / `corrective RAG` / `self-reflective retrieval` / `generative relevance feedback` / `pseudo-document` / `query understanding` / `multi-query` / `underspecified query`。

再按 arXiv ID 补入 12 篇经典工作（`scripts/add_curated.py`）。

### 3.2 引用数校准
- Semantic Scholar 批量接口匿名限流严重（持续 429），**不可用**；
- OpenAlex 的 arXiv DOI 覆盖不全（HyDE 的 arXiv DOI 查不到，换标题搜才拿到 371）；
- 最终方案：按标题做 OpenAlex 搜索，`difflib` 相似度 > 0.85 才采纳（`scripts/enrich_cites.py`）；
- 结果：300 个候选中 103 个拿到引用数。

### 3.3 下载与解析
- `scripts/download_parse.py`：requests 下载 → 校验 `%PDF` 头 → `pymupdf4llm.to_markdown()`；
- 每篇 Markdown 顶部写入 YAML frontmatter（arxiv/title/year/cites/source）；
- 23/23 全部成功，无缺失。

### 3.4 复现命令

```bash
cd /disk1/rag/query-evidance
export http_proxy="http://<USER>:<PASS>@10.251.112.50:8128"
export https_proxy="$http_proxy"

python3 scripts/enrich_cites.py      # 可断点续跑，已有 oa_cites 的会跳过
python3 scripts/add_curated.py       # 幂等，按 arXiv ID 去重
python3 scripts/download_parse.py selected.json
```

---

## 4. 数据质量与已知缺陷

1. **`candidates.json` 中有 3 个错误的 ID 映射**（人工补入时 ID 记错，均未被选入 23 篇）：
   - `2212.09477` 实际是一篇数论论文，不是 ConvGQR；
   - `2401.17043` 实际是 CRUD-RAG 基准，不是 LameR；
   - `2310.08559` 实际是归纳推理论文，不是 Self-Knowledge Guided Retrieval。
   如后续要引用这三篇，请重新核对 ID。
2. **引用数偏差**：OpenAlex 对 arXiv 版本覆盖不全，部分论文引用数偏低（如 HyDE 在 Google Scholar 远高于 371）。
3. **Markdown 解析噪声**：复杂表格会被压成单行，公式可能丢失定界符。**引用具体数字前必须回看 PDF**（`papers/`）。
4. **检索不穷尽**：只覆盖 15 组查询，2026 年下半年新工作可能有遗漏。
5. **综述中的数字已抽验**：HyDE、Query2Doc、doc2query、IRCoT、Adaptive-RAG、CRAG、RARR、Least-to-Most 的关键数字均与解析正文核对过；Self-RAG 的引用精度结论来自原文摘要与结论段。

---

## 5. 研究方案：SALR（Slot-Aligned Retrieval）

### 5.1 动机（为什么是"直接对齐"）

现有 RAG 优化的是 $\max_d \text{sim}(q,d)$，但相似度衡量的是**主题相关**，不是**语义角色对齐**。两类典型失败：

- **高相似、零覆盖**：证据主题相关，但没有回答 query 要的槽位；
- **低相似、全覆盖**：正确证据换了措辞（head vs director），被相似度排到后面。

因此把检索目标从"相似度最大"改为"**信息需求被证据覆盖最多**"。

### 5.2 形式化

**Query → 槽位框架**（一次 LLM 调用，JSON）：

```json
{
  "answer_type": "person",
  "entities": ["Parasite", "Palme d'Or", "2019"],
  "slots": [
    {"id": "s1", "relation": "award_year->film", "head": "Palme d'Or", "year": "2019"},
    {"id": "s2", "relation": "film->director", "head": "$s1"}
  ],
  "constraints": []
}
```

**证据 → 命题集合**（对 top-N 候选抽取，按 doc_id 缓存）：

```json
{"s": "Parasite", "r": "directed_by", "o": "Bong Joon-ho", "span": "…directed by Bong Joon-ho…", "type": "person"}
```

**对齐分数**：

$$
A(q,e) = \sum_{s \in \text{slots}(q)} w_s \cdot \max_{p \in P(e)} \big[ \text{rel}(s,p) \cdot \text{ent}(s,p) \cdot \text{type}(s,p) \big]
$$

- `rel`：关系归一化（head ≈ director ≈ directed by），别名词表 + LLM 归一；
- `ent`：实体匹配（别名、缩写、全称）；
- `type`：答案类型兼容。

**检索**：$\arg\max_e A(q,e)$；相似度仅作 tie-break 与兜底。

**未覆盖槽位驱动下一跳**：若某槽位覆盖为 0，直接用该槽位实例化查询：

$$
q' = \text{render}(\text{head},\; r,\; \text{answer\_type})
$$

例：`s2` 未覆盖 → `"Parasite director"`。**不是自由生成子问题，而是由对齐缺口直接生成。**

**停止条件**（可验证）：所有槽位都有匹配命题，且答案类型匹配。

### 5.3 一个可写进论文的例子

Query：*Who directed the film that won the Palme d'Or in 2019?*

- 候选 A：*"Parasite is a 2019 South Korean film directed by Bong Joon-ho. It won the Palme d'Or."*
  → (Parasite, won, Palme d'Or)、(Parasite, directed_by, Bong Joon-ho)、(Parasite, year, 2019)
  → 槽位覆盖 **2/2**
- 候选 B：*"The 2019 Palme d'Or ceremony was held in Cannes…"*
  → (ceremony, held_in, Cannes)、(ceremony, year, 2019)
  → 槽位覆盖 **1/2**

BM25 往往把 B 排前（"Palme d'Or 2019" 词面重合多），SALR 把 A 提到第一。这是最直观的"直接对齐 vs 相似度"对照实验。

### 5.4 与现有工作的区别

| 方法 | 对齐发生在 | 依据 |
|---|---|---|
| BM25 / 稠密检索 | 表示层 | 词面或向量相似 |
| HyDE / Query2Doc | query 侧 | 改写成"像文档"的文本，仍靠相似度 |
| IRCoT | 迭代检索 | CoT 文本当 query，无显式对齐 |
| CRAG | 检索后 | 文档级相关性判定 |
| **SALR** | **检索时** | **槽位—命题结构化匹配，覆盖率即检索目标** |

### 5.5 被否决的两个方案（避免重复讨论）

| 方案 | 内容 | 否决原因 |
|---|---|---|
| **GASA** | 子问题级 gap 诊断 + 类型化修复算子 + 反事实监督 | 本质是"失败后修复"，动机不够直接；与"先分解、再验证、失败兜底"的一类框架在思路上重叠，增量不足 |
| **GRIP** | 在验证信号触发后做 gap 类型诊断，选择 4 类算子 | 同上：仍是后置修复，用户明确要求"更直接地对齐" |

> 保留价值：GASA/GRIP 的算子库（关系扩展、假设答案、再分解、实体枢轴）可作为 SALR 的**兜底策略**，但不是主贡献。

---

## 6. 实验协议

### 6.1 数据
| 数据集 | 用途 | 规模 |
|---|---|---|
| HotpotQA（distractor validation） | 主实验 | 200 题，seed=42 |
| 2WikiMultiHopQA（dev） | 主实验 | 200 题，seed=42 |
| MuSiQue（dev） | 主实验（最难） | 200 题，seed=42 |
| Bamboogle | 组合性专项（可选） | 125 题全量 |
| Natural Questions / TriviaQA | 单跳不退化（可选） | 各 200 题 |

> 三个主数据集使用**同一批题**跑所有方法，保证配对比较。

### 6.2 检索
- BM25Okapi（`rank_bm25`），小写 + 空格分词，`k1=1.5, b=0.75`；
- 首次检索 top-10，最终取 top-5；
- 语料：
  - HotpotQA：distractor 设定的 10 段上下文（简单、快），或全量 Wikipedia 段落（更真实但更大）；
  - 2Wiki / MuSiQue：各数据集自带上下文，构成本地语料池；
  - **pilot 阶段一律用各数据集自带上下文**，避免下载全量 Wikipedia。

### 6.3 模型
`qwen3.6-27b`，`temperature=0`，`enable_thinking=False`；两端点轮询 + 失败切换 + 结果缓存。

### 6.4 方法
**基线**：Vanilla RAG、Self-Ask、IRCoT、ReAct、BM25+Cross-encoder 重排（可选）。
**新方法**：
- `SALR-rerank`：仅用对齐分数重排 BM25 top-10；
- `SALR+Self-Ask`：每个子问题用 SALR 检索；
- `SALR-full`：槽位对齐 + 未覆盖槽位驱动下一跳 + 可验证停止。
**消融**：
1. 去掉关系归一化；
2. 去掉类型约束；
3. 去掉实体归一化；
4. 去掉未覆盖槽位驱动（改回固定 top-k）；
5. oracle 槽位 vs LLM 解析槽位（量化解析误差的代价）。

### 6.5 指标
- 检索层：金标证据 Recall@5/10、**槽位覆盖率**、对齐分数与正确性的相关性（Spearman）；
- 端到端：EM、F1（SQuAD 归一化）；
- 专项：**关系同义词子集**（head/director、born in/native of 等）上的 EM 提升；
- 效率：LLM 调用数、prompt/completion token、端到端时延；
- 稳健性：槽位解析失败率、命题 span 校验失败率。

### 6.6 成本估算（pilot）
- 600 题 × 1 次槽位解析 ≈ 600 次调用；
- 600 题 × 10 候选命题抽取 ≈ 6,000 次调用（按 doc_id 缓存后显著减少）；
- 单次约 0.1–1 s，pilot 预计 1–2 小时（不含重试）。

### 6.7 验收标准（pilot）
| 指标 | 通过线 |
|---|---|
| 槽位解析成功率 | ≥ 90% |
| 命题 span 逐字校验通过率 | ≥ 85% |
| 金标证据 Recall@5 | 相对 BM25 提升 ≥ 10 个点 |
| 关系同义词子集 | 有肉眼可见改善（至少 3 个案例） |
| 额外延迟 | 单题 < 3 s（pilot 规模） |

---

## 7. 隔离环境从零搭建（Runbook）

### 7.1 目录结构

```
<NEW_ROOT>/
├── data/
│   ├── raw/            # 原始数据集（json/parquet）
│   ├── corpus/         # 每数据集一个 passages.jsonl（id -> text）
│   └── index/          # bm25_*.pkl
├── src/
│   ├── llm.py          # 端点轮询 / 重试 / 缓存 / token 统计
│   ├── retriever.py    # BM25 构建与检索
│   ├── frame.py        # query -> 槽位框架
│   ├── propositions.py # 证据 -> 命题
│   ├── align.py        # 对齐打分与重排
│   ├── pipeline.py     # SALR / Self-Ask / IRCoT 主流程
│   └── eval.py         # EM/F1/Recall/bootstrap CI
├── cache/              # LLM 缓存（sqlite）
├── results/
├── logs/
└── configs/            # 冻结的协议参数
```

### 7.2 数据获取

```python
from datasets import load_dataset

hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
wiki2  = load_dataset("voidful/2WikiMultihopQA")          # 或官方 json
musique = load_dataset("dgslibisey/MuSiQue")               # 或官方 jsonl
```

> 若 HF 不可达，用代理；或从官方 GitHub 下载 json/jsonl。**pilot 用 distractor/自带上下文即可，不要下载全量 Wikipedia。**

### 7.3 BM25

```python
from rank_bm25 import BM25Okapi
tokenized = [doc.lower().split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
```

- 语料按 passage 粒度切分；
- 保存 `(doc_ids, corpus_texts, bm25)` 到 pickle；
- 注意内存：31 GB 机器上 6 万段没问题，百万级需要分片。

### 7.4 LLM 客户端要点
1. 两个端点轮询，失败自动切换；
2. 按 `sha1(model + messages + params)` 缓存到 sqlite；
3. 统计真实调用数、缓存命中、token；
4. 强制 `enable_thinking=False`；
5. JSON 输出加 `response_format={"type":"json_object"}`（若端点支持），否则用严格 prompt + 解析失败重试 2 次。

### 7.5 评测实现
- EM/F1 用 SQuAD 官方归一化（小写、去标点、去冠词、数字转文字）；
- 多答案取 max；
- 配对 bootstrap（1000 次）给 95% CI；
- 所有方法共用同一批题、同一模型、同一检索器、同一 k。

---

## 8. Prompt 与 JSON Schema

### 8.1 槽位解析

```
你是信息检索规划器。把问题解析成结构化槽位，只输出 JSON。
规则：
1. answer_type 只能是 person/date/number/location/organization/title/boolean/other 之一；
2. slots 按求解顺序排列，后一个槽位可以用 "$sX" 引用前一个槽位的答案；
3. relation 用蛇形命名（如 directed_by、award_year_to_film）；
4. 不要编造实体；实体必须出现在原问题中。

问题：{question}
输出：{"answer_type": "...", "entities": [...], "slots": [...], "constraints": [...]}
```

### 8.2 命题抽取

```
从下面段落中抽取原子命题，只输出 JSON 数组。
规则：
1. 每条命题是 (s, r, o)，r 用蛇形命名；
2. span 必须是原文中逐字出现的片段，不得改写；
3. 只抽取与段落主旨相关的事实，最多 8 条；
4. 无法确定时省略，不要猜测。

段落：{passage}
输出：[{"s": "...", "r": "...", "o": "...", "span": "...", "type": "..."}]
```

### 8.3 对齐判定（可选，用于难例复核）

```
给定槽位和候选命题，判断是否语义匹配。
只输出 JSON：{"match": true/false, "reason": "..."}
槽位：{slot}
命题：{proposition}
```

---

## 9. 已知坑与规避

| 坑 | 现象 | 规避 |
|---|---|---|
| curl TLS | 对 arXiv/Google 报 `NSS error -12190` | 用 Python `requests` |
| Semantic Scholar 限流 | 持续 429 | 改用 OpenAlex 标题搜索 |
| OpenAlex DOI 缺失 | arXiv DOI 查不到引用 | 按标题搜 + 相似度过滤 |
| 内网 API 走代理 | 请求超时 | `proxies={}` 或 `--noproxy '*'` |
| Qwen 思考模式 | 输出被 "thinking process" 占满 | `enable_thinking=False` |
| 多写者改文件 | 编辑被 stale-context 拒绝 | 单进程/单写者；批量改动用脚本一次性写入 |
| JSON 解析失败 | LLM 输出带 markdown 代码块 | 剥离 ```json 围栏 + 重试 + 失败回退 |
| BM25 内存 | 31 GB 机器加载大索引 OOM | 分片或改用 distractor 小语料 |
| 数据污染 | HotpotQA/2Wiki 可能被预训练覆盖 | 用 Bamboogle/MuSiQue 交叉验证；报告时注明 |

---

## 10. 待决策事项

1. **语料规模**：pilot 用数据集自带上下文（快），正式实验是否上 HotpotQA 全量 Wikipedia？
2. **关系归一化方式**：纯别名词表 vs LLM 归一 vs 两者级联（当前建议：级联，LLM 只处理未命中）。
3. **命题抽取粒度**：句级 vs 段落级（当前建议：句级，span 更易校验）。
4. **是否加入单跳不退化实验**（NQ/TriviaQA）。
5. **是否把 SALR 与其它分解/执行框架叠加**（可作为后续工作，不作为主线）。

---

## 11. 交接检查清单

- [x] 23 篇 PDF 与 Markdown 齐全
- [x] 综述、Top10 精读、索引、元数据齐全
- [x] 引用数校准方法与缺陷已记录
- [x] 环境/网络/API 约束已记录
- [x] 研究方案（SALR）与实验协议已记录
- [x] 被否决方案与原因已记录
- [x] 从零搭建步骤、Prompt、Schema 已记录
- [ ] 数据与 BM25 索引（隔离环境需重建）
- [ ] 20 题 pilot（未执行）
- [ ] 全量 3×200 实验（未执行）

---

## 12. 附录：关键数字速查（引用前请回看 PDF）

| 论文 | 设置 | 结果 |
|---|---|---|
| HyDE | DL19 / DL20 nDCG@10 | 44.5→61.3 / 42.1→57.9 |
| Query2Doc | TREC DL19/20，BM25 | 相对提升 >15% |
| doc2query | TREC-CAR / MS MARCO，BM25 | 约 +15% |
| IRCoT | 多跳召回 | +7.9–22.6 点；F1 最高 +15 |
| Adaptive-RAG | FLAN-T5-XL 单跳 F1 | 44.31→46.94 |
| CRAG | SelfRAG-LLaMA2-7B 底座 | PopQA +7.0；Bio +14.9 FactScore；ARC +36.6 |
| RARR | NQ/SQA/QReCC F1AP | 57.0 / 45.9 / 41.5（EFEC：17.1 / 7.1 / 31.9） |
| Least-to-Most | GSM8K ≥5 步 | 39.07→45.23（整体 60.97→62.39） |
