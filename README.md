# Binding, not Matching：重新诊断 RAG 的 query–证据语义 gap

围绕 **"RAG 中 query 与检索证据的语义 gap"** 的完整工作：文献调研 → 机制验证 → 规模化实验 → 诊断修正。

## 一句话

语义 gap 主要不是"query 和证据不够像"，而是**query 里的需求变量没有被绑定**。绑定后第一段召回按跳次崩塌的现象消失；剩余损失可分解为：

$$\text{gap 消除率}=\underbrace{\text{候选可达性}}_{\text{绑定/第一段}}\times\underbrace{\text{匹配能力}}_{\text{CE/蒸馏}}$$

## 先读什么

| 文档 | 内容 |
|---|---|
| [`docs/01_PROCESS.md`](docs/01_PROCESS.md) | **过程记录**：按时间顺序的全部实验、决策、18 个 bug 与修复、被推翻的结论 |
| [`docs/02_PROJECT.md`](docs/02_PROJECT.md) | **项目文档**：背景、形式化、方法（PAG）、全部参数、目录与运行 |
| [`docs/03_EXPERIMENTS.md`](docs/03_EXPERIMENTS.md) | **实验文档**：假设、E0–E11 结果、消融、失败分析、有效性威胁 |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | 研究方向、与已有工作的边界、逐阶段结果 |
| [`PILOT.md`](PILOT.md) | 20 题 pilot 的独立报告 |
| [`HANDOVER.md`](HANDOVER.md) | 最初的交接文档（环境、调研方法、SALR 方案） |
| [`survey.md`](survey.md) / [`top10.md`](top10.md) | 23 篇论文的综述与 Top10 精读 |

## 三个贡献

- **C1 诊断**：把 gap 从"表示不匹配"重述为"绑定缺失 + 选择错位"，给出直接度量。
- **C2 方法**：plan-conditioned grounding（计划 → 绑定 → 逐需求 CE → 覆盖选择 → 可验证停止）。
- **C3 负结果**：查询改写只消 9.5% 的 gap；LLM 绝对充分性判定不是上界。

## 关键数字

| 实验 | 结果 |
|---|---|
| 绑定因果实验 | 按跳召回 0.915→0.538；绑定后 **0.92–0.97**（同语料、同标签） |
| 端到端（MuSiQue 200） | ours **0.425 EM** vs IRCoT 0.330（Δ+0.095 [+0.040,+0.150]） |
| 端到端（MuSiQue 400 同池） | ours **0.430** vs IRCoT 0.255（Δ+0.175 [+0.130,+0.223]） |
| 端到端（2Wiki 200） | ours **0.605** vs IRCoT 0.535（Δ+0.070 [+0.010,+0.125]） |
| 端到端（HotpotQA 200） | ours 0.660 vs IRCoT 0.660（打平） |
| 2×2 消融 | 绑定 +0.14、CE 选证 +0.08，近乎可加；绑定单独只追平 IRCoT |
| 语义 gap 占比（chunk 级） | **17.6%**（dense）/ 23.6%（BM25） |
| gap 消除 | 改写 9.5% / 陈述句 31.6% / **CE 35.8%** |
| 真实口径统一对照（582 case） | base CE 0.198 / 蒸馏 0.227 / LLM 0.170 / BM25 0.081（closure@3） |

## 代码

```
src/           实验代码（llm/ce_client/data/retriever/eval + E0–E11）
results/       全部原始结果（jsonl/json/md）
data/raw/      MuSiQue / HotpotQA 题目缓存
deploy/        cross-encoder 服务
dist/          E7–E11 GPU 训练包（难例挖掘 + 蒸馏）
```

运行方式见 [`docs/02_PROJECT.md`](docs/02_PROJECT.md) §10。长任务均按题增量写盘，可断点续跑。

## 局限

- 端到端增益有限（MuSiQue +0.075，HotpotQA 打平）。
- 匹配器蒸馏在真实口径只 +2.9 点；LLM 教师不是上界。
- 仅英文 Wikipedia 两个数据集。
- "可达性 × 匹配能力"仍是经验分解，没有正式命题。

## 附：调研产物

- `papers/`：23 篇 arXiv PDF（28 MB，未入 git）
- `markdown/`：PyMuPDF4LLM 解析文本（未入 git）
- `papers.csv` / `selected.json` / `candidates.json`：元数据与筛选记录
- `scripts/`：引用数校准、补充经典论文、下载与解析

引用数来自 OpenAlex（2026-11-04），普遍低于 Google Scholar，仅用于排序。
