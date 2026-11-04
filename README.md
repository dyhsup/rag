# RAG Query–证据语义 gap：调研、测量与消除

围绕 **"RAG 中 query 与检索证据的语义 gap"** 的完整工作：文献调研 → 机制验证 → 规模化实验 → 方法。

## 先读什么

| 文档 | 内容 |
|---|---|
| [`docs/01_PROCESS.md`](docs/01_PROCESS.md) | **过程记录**：按时间顺序的全部实验、决策、13 个 bug 与修复、被推翻的结论 |
| [`docs/02_PROJECT.md`](docs/02_PROJECT.md) | **项目文档**：背景、问题形式化、方法（PAG）、全部参数、目录与运行方式 |
| [`docs/03_EXPERIMENTS.md`](docs/03_EXPERIMENTS.md) | **实验文档**：假设、设置、E0–E7 结果表、消融、失败分析、有效性威胁 |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | 研究方向、与已有工作的边界、逐阶段结果 |
| [`PILOT.md`](PILOT.md) | 20 题 pilot 的独立报告 |
| [`HANDOVER.md`](HANDOVER.md) | 最初的交接文档（环境、调研方法、SALR 方案） |
| [`survey.md`](survey.md) / [`top10.md`](top10.md) | 23 篇论文的综述与 Top10 精读 |

## 一句话结论

语义 gap 在 chunk 级语料上占 **17.6%** 的检索失败（证据召回得到、但排不进 top-3）。它可分解为两个乘性因子：

$$\text{gap 消除率}=\underbrace{0.674}_{\text{候选可达性}}\times\underbrace{0.734}_{\text{匹配能力}}=0.495$$

- **可达性**：后几跳证据的召回从 0.915 掉到 0.538，因为查询里没有中间实体；绑定后回到 **0.89–0.97**。
- **匹配能力**：查询侧改写只消除 **9.5%** 的 gap；现成 cross-encoder 消除 **35.8%**，LLM 判定 **49.5%**（可达子集 **73.4%**）。

## 主要结果

| 实验 | 结果 |
|---|---|
| 端到端（MuSiQue，200 题） | ours **0.405 EM** vs IRCoT 0.330（Δ+0.075 [+0.020,+0.130]） |
| 端到端（HotpotQA，200 题） | ours 0.675 vs IRCoT 0.660（Δ+0.015，打平） |
| 2×2 消融 | 绑定 +0.14、CE 选证 +0.08，近乎可加；绑定单独只追平 IRCoT |
| 绑定因果实验 | 召回 0.54–0.67 → **0.89–0.97**（同语料、同标签） |
| 语义 gap 测量 | 17.6%（dense）/ 23.6%（BM25） |
| gap 消除 | 改写 9.5% / 陈述句 31.6% / **CE 35.8%** / LLM 上界 49.5% |

## 代码

```
src/           实验代码（E0–E7）
results/       全部原始结果（jsonl/json/md）
data/raw/      MuSiQue / HotpotQA 题目缓存
deploy/        cross-encoder 服务
dist/e7/       GPU 训练包（蒸馏匹配器）
```

运行方式见 [`docs/02_PROJECT.md`](docs/02_PROJECT.md) §10。长任务均按题增量写盘，可断点续跑。

## 局限

- 仅英文 Wikipedia 两个数据集；2Wiki 镜像不可用。
- 读者（top-3/top-8）是端到端上限。
- E7 蒸馏结果待回传。
- 理论只有经验曲线与草稿，没有正式命题。

## 附：调研产物

- `papers/`：23 篇 arXiv PDF（28 MB，未入 git）
- `markdown/`：PyMuPDF4LLM 解析文本（未入 git）
- `papers.csv` / `selected.json` / `candidates.json`：元数据与筛选记录
- `scripts/`：引用数校准、补充经典论文、下载与解析

引用数来自 OpenAlex（2026-11-04），普遍低于 Google Scholar，仅用于排序。
