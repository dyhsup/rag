# RAG 中 Query–证据语义不匹配：问题、方法与进展综述

> 调研范围：检索增强生成（RAG）中"用户 query 与检索到的证据在语义上对不上"这一核心问题，及其缓解方法。
> 语料：本目录 `papers/` 下 23 篇论文（PDF），`markdown/` 为对应解析文本。
> 引用数为 OpenAlex 统计（截至 2026-11-04），普遍低于 Google Scholar，仅用于相对排序。

---

## 0. 结论先行

1. **不匹配不是单点问题，而是四类失配的叠加**：词面/词汇失配、意图与粒度失配、表示与分布失配、证据有效性失配。不同方法实际上在修不同的一层，横向比较时必须先对齐"修的是哪一层"。
2. **最有效且最通用的两类手段**是：(a) 在 query 侧生成"假设性证据"或改写 query，把 query 拉到证据的分布里（HyDE、Query2Doc、Rewrite-Retrieve-Read）；(b) 把复杂 query 分解成与证据粒度匹配的子问题并迭代检索（Least-to-Most、Self-Ask、IRCoT）。
3. **生成式扩展的红利随检索器变强而递减**。Query2Doc 明确观察到：对 BM25 提升 >15%，但对经过中间预训练/蒸馏的稠密检索器增益明显变小。这意味着"补语义鸿沟"与"提升检索器本身"存在部分替代关系。
4. **检索后纠错（CRAG、Self-RAG、RARR）解决的是另一类失配**——检索回来了但证据不可用/不被忠实使用。它们的上限由"判定器"的可靠性决定：CRAG 的实验显示轻量 T5 evaluator 反而优于 GPT-3.5 提示判定。
5. **工程上最划算的组合**通常是：轻量 query 复杂度路由（Adaptive-RAG）+ 假设文档扩展（HyDE/Query2Doc）+ 多跳时迭代检索（IRCoT）+ 生成后归因校验（RARR/CRAG）。代价是延迟与 token 成本，需要按场景裁剪。

---

## 1. 问题界定

### 1.1 什么是 query–证据语义不匹配

标准 RAG 用同一个相似度函数比较 query 与文档块：

$$
s(q, d) = \langle f(q),\, g(d) \rangle
$$

不匹配（mismatch / gap）指：**在语义上足以回答 query 的证据 $d^\*$，其表示 $g(d^\*)$ 与 $f(q)$ 的相似度并不高**，因而在 top-k 中缺席或被噪声挤占。它表现为两种失败：

- **召回失败**：正确证据根本没进 top-k；
- **排序失败**：正确证据进了候选但排在噪声之后，生成阶段被"中间丢失"或被误导。

### 1.2 四类失配来源

| 类型 | 典型表现 | 例子 |
|---|---|---|
| **词面/词汇失配** (lexical) | query 与证据用词不同、缩写/同义/跨语言 | 问"心肌梗死"，文档写"AMI"；BM25 直接失效 |
| **意图与粒度失配** (intent & granularity) | query 是疑问句、多跳、含隐含约束；证据是陈述句、单跳片段 | "A 的导师的博士母校是？" 需要两段证据链；单段证据只覆盖一半 |
| **表示/分布失配** (representation) | query 短、口语、带指代；文档长、书面、含表格 | 对话中的"那它呢？"；短 query 与长 chunk 的向量不对称 |
| **证据有效性失配** (validity) | 检索到了相关内容，但过时、错误、或与问题实体不一致 | 问 2026 年数据，检索到 2021 年文档；生成时未忠实使用证据 |

前两类是**检索前**问题（query 侧、索引侧），第三类是**表示层**问题，第四类是**检索后**问题。这个划分对应第 2 节的方法分类。

### 1.3 为什么 RAG 会放大不匹配

- **非对称编码**：query 与 document 由不同编码器（或同一编码器的不同分布）映射；训练目标只保证"相关对"靠近，不保证"问题—答案句"靠近。
- **chunk 化破坏上下文**：答案句常被切在 chunk 边界，指代与限定条件丢失。
- **单向量瓶颈**：一个向量要同时表达实体、关系、约束、时间，信息被压缩。
- **检索器训练分布与线上 query 分布不一致**：长尾实体、领域术语、多语言 query 上尤其明显。

---

## 2. 方法分类

按"在流水线的哪一环修补"划分：

```
query ──► [A. query 侧变换] ──► retriever ──► [C. 检索后校验/纠错] ──► generator
              │                      ▲
              │                      │
              └── [B. 索引/证据侧扩展] ┘
              
         [D. 控制层：按 query 复杂度选择上述策略]
```

### A. Query 侧变换

#### A1. 假设文档 / 生成式扩展（把 query 变成"像文档的东西"）

| 方法 | 核心机制 | 引用 |
|---|---|---|
| **HyDE** | 让指令模型先生成一段"假设答案文档"，再用无监督对比编码器编码该文档去检索；生成内容的错误细节被编码器瓶颈过滤 | 371 |
| **Query2Doc** | 少样本提示 LLM 生成伪文档，把 `query + 伪文档` 拼接作为新 query；对 BM25 与稠密检索都适用，无需训练 | 220 |
| **GRF** (Generative Relevance Feedback) | 不依赖首轮检索结果做伪相关反馈，而是用 LLM 直接生成"相关性反馈模型"，避免首轮结果全错时 PRF 失效 | 49 |
| **HyPE** (Hypothetical Prompt Embeddings) | 预计算"假设性提示嵌入"，把运行时生成搬到离线，降低 HyDE 的在线延迟 | 4 |
| **MILL** | 生成式扩展与检索式扩展互相验证，缓解单一扩展来源的意图漂移 | 17 |

**共同假设**：证据空间是"文档分布"，query 是"问题分布"；把 query 投到文档分布里，相似度才有意义。
**共同风险**：伪文档会引入错误实体；对已经强监督的检索器增益递减；在线生成带来延迟。

#### A2. Query 改写与多查询

| 方法 | 核心机制 | 引用 |
|---|---|---|
| **Rewrite-Retrieve-Read** | 用 LLM 改写 query 再检索；进一步用小型 T5 rewriter，以 reader 的 EM/F1 为奖励做 PPO 训练，使 query 对齐下游 reader | 276 |
| **RaFe** | 用排序反馈（而非人工标注/reader 奖励）训练 rewriter，降低 RL 奖励稀疏问题 | 21 |
| **Crafting the Path** | 针对 Query2Doc 类方法在"LLM 内部知识不足"时失效的问题，约束改写路径、减少幻觉扩展 | 4 |
| **DMQR-RAG** | 多查询、多样化改写，覆盖 query 的多种意图解释；配合相关性感知的保留策略 | 2 |
| **TreeHop** | 把多跳改写放到 embedding 层做轻量路由，减少反复调用 LLM 的开销 | 0 |

**要点**：改写解决的是"原 query 不是好的检索 query"（含噪声、口语、隐含意图）。训练信号从"人工标注 query"演进到"reader 反馈"再到"排序反馈"。

#### A3. Query 分解

| 方法 | 核心机制 | 引用 |
|---|---|---|
| **Least-to-Most** | 两阶段提示：先分解子问题，再按难度递增依次求解；后一个子问题依赖前一个答案 | 388 |
| **Self-Ask** | 模型显式生成并回答后续子问题；可外接搜索引擎逐个回答子问题 | 303 |
| **IRCoT** | 交替进行"生成一句 CoT"与"用该句检索"，检索由推理引导、推理又被检索修正 | 359 |

**解决的是粒度失配**：复杂 query 需要多条证据，而单次检索只能返回"与整句相似"的片段。分解把 query 变成与证据粒度匹配的原子问题。

### B. 索引/证据侧扩展

| 方法 | 核心机制 | 引用 |
|---|---|---|
| **doc2query** | 训练 seq2seq 为每篇文档预测"用户可能怎么问"，把预测 query 追加到文档再建索引（文档扩展） | 215 |
| **Knowledge-Aware Query Expansion** | 同时利用文本相似与文档间关系（图/关系）做扩展，缓解只看文本相似导致的漏召回 | 5 |
| **Semantic Models for First-stage Retrieval（综述）** | 系统梳理稀疏/稠密/学习型稀疏第一段检索的语义建模，是理解表示层失配的基础读物 | 119 |

**要点**：doc2query 是"在索引时把文档改写成 query 的样子"，与 HyDE"在查询时把 query 改写成文档的样子"互为镜像。二者结合即"双向对齐"。

### C. 检索后校验与纠错

| 方法 | 核心机制 | 引用 |
|---|---|---|
| **CRAG** | 轻量 T5 evaluator 给检索结果打置信度 → {Correct, Incorrect, Ambiguous} 三路动作：正确则分解-过滤-重组知识条；错误则丢弃并转 web 搜索；不确定则两者融合 | 187 |
| **Self-RAG** | 训练模型生成反思 token（是否检索 Retrieve、证据是否相关 ISREL、是否支持 ISSUP、是否有用 ISUSE），推理时可控地按需检索与引用 | 104 |
| **RARR** | 对任意生成模型的输出做事后归因：生成问题→检索证据→回答问题→比对，再由编辑器最小化改写不支持片段 | 148 |

**解决的是证据有效性/忠实性失配**。与 A/B 类方法的区别：它们不改变"检索什么"，而是判断"检索到的东西能不能用、怎么用"。

### D. 控制层

| 方法 | 核心机制 | 引用 |
|---|---|---|
| **Adaptive-RAG** | 训练 query 复杂度分类器（简单/单步/多步），动态选择"不检索 / 单步检索 / 多步迭代检索" | 264 |
| **Query Optimization 综述** | 把 query 优化系统化为 query 生成、改写、分解、消歧等模块 | 2 |

**洞察**：不匹配的程度因 query 而异。对所有 query 都用重型多跳流水线既慢又可能引入噪声；路由是性价比最高的工程手段。

---

## 3. 方法对比

| 方法 | 作用位置 | 是否需训练 | 额外成本 | 主要修补的失配 | 代表评测 |
|---|---|---|---|---|---|
| Least-to-Most | query 侧/推理侧 | 否（提示） | 多次 LLM 调用 | 粒度（多步） | GSM8K, DROP, SCAN |
| Self-Ask | query 侧/推理侧 | 否 | 多次调用，可外接搜索 | 粒度（组合性） | Bamboogle, 2Wiki, MuSiQue |
| IRCoT | query 侧 + 迭代检索 | 否 | 每步检索 | 粒度 + 检索时机 | HotpotQA, 2Wiki, MuSiQue, IIRC |
| HyDE | query 侧（表示） | 否 | 1 次生成 + 编码 | 词面/风格/表示 | DL19/20, BEIR, Mr.TyDi |
| Query2Doc | query 侧（表示） | 否 | 1 次生成 | 词面/风格 | MS MARCO, DL19/20, BEIR |
| GRF | query 侧（反馈） | 否 | 生成反馈模型 | 首轮检索不可靠 | TREC DL, MS MARCO |
| Rewrite-Retrieve-Read | query 侧 | 可选（RL rewriter） | 改写 + 检索 | 意图/噪声 | HotpotQA, AmbigNQ, MMLU |
| RaFe | query 侧 | 是（排序反馈） | 训练成本 | 意图/噪声 | 开放域 QA |
| doc2query | 索引侧 | 是（seq2seq） | 离线索引膨胀 | 词面（双向） | MS MARCO, TREC-CAR |
| CRAG | 检索后 | 轻量 evaluator | 判定 + 可能 web 搜索 | 证据有效性 | PopQA, Bio, PubHealth, ARC |
| Self-RAG | 生成侧 | 是（critic + 反思 token） | 训练 + 推理 | 是否检索/证据是否支持 | 6 任务（QA/事实验证/长文） |
| RARR | 生成后 | 编辑器（少量样本） | 检索 + 归因 + 改写 | 忠实性/归因 | ASQA, NQ, QReCC, SQA |
| Adaptive-RAG | 控制层 | 是（分类器） | 分类开销 | 按复杂度分配策略 | SQuAD, NQ, TriviaQA, MuSiQue, HotpotQA, 2Wiki |

---

## 4. 关键实验证据

> 以下数字均取自各论文正文/表格，标注了出处论文。

- **HyDE**（DL19/DL20，nDCG@10）：Contriever 44.5 → **61.3**；DL20 42.1 → **57.9**，已接近有标注微调的 Contriever[FT]（62.1/63.2）。
- **Query2Doc**（BM25）：TREC DL 2019/2020 上相对 BM25 提升 **>15%**；但作者明确指出，检索器越强（中间预训练、蒸馏），增益越小。
- **doc2query**（BM25）：TREC-CAR 与 MS MARCO 上检索效果提升约 **15%**；与 RM3 组合并不总是正向叠加。
- **IRCoT**：多跳检索召回相对单步检索提升 **7.9–22.6** 个点（视数据集/模型）；端到端 F1 最高提升约 **15** 点；3B 的 Flan-T5 + IRCoT 可超过大 58 倍的 GPT-3 单步检索。
- **Adaptive-RAG**（FLAN-T5-XL 3B，单跳）：单步检索 F1 44.31 → **46.94**，同时平均检索步数与耗时低于固定多步策略。
- **CRAG**（以 SelfRAG-LLaMA2-7B 为底座）：相对标准 RAG，PopQA 准确率 **+7.0**，Biography FactScore **+14.9**，ARC-Challenge **+36.6**。
- **Self-RAG**（7B/13B）：在 PubHealth、PopQA、ASQA 等任务上超过 ChatGPT 与检索增强的 Llama2-chat，并显著提升引用精度。
- **RARR**：在 NQ/SQA/QReCC 上，F1AP 分别为 **57.0 / 45.9 / 41.5**，显著高于微调编辑器 EFEC（17.1 / 7.1 / 31.9），且在保持原文（preservation 80–90）的前提下大幅提升归因；模型无关，可直接套在任意生成模型输出上。
- **Least-to-Most**：GSM8K 从 60.97 → 62.39，但在**需要 ≥5 步**的题目上从 39.07 → **45.23**，说明收益集中在长推理链。

---

## 5. 规律与反直觉发现

1. **"补语义"与"换更强检索器"部分互斥**。生成式扩展本质是在用 LLM 的世界知识弥补检索器表示能力的不足；当检索器本身足够强（领域预训练、蒸馏），外部扩展的边际收益下降。
2. **多跳场景的瓶颈往往不是编码器，而是"检索时机"**。IRCoT 表明：用整句 query 一次检索，后续推理步所需的证据不在 top-k；按推理步迭代检索收益巨大。
3. **伪相关反馈在首轮全错时会放大错误**。GRF 的动机正是这一点：不依赖首轮结果，改为生成反馈模型。
4. **纠错模块的上限由判定器决定**。CRAG 用轻量 T5 evaluator 反而优于 GPT-3.5 提示判定——小模型在"相关性二分类"上更稳定、更便宜。
5. **改写需要与下游 reader 对齐**。Rewrite-Retrieve-Read 用 reader 的 EM/F1 做 RL 奖励，说明"好的检索 query"不等于"人类觉得自然的 query"。
6. **自适应路由能同时改善效果与效率**，但分类器精度是瓶颈；论文也把"更好的复杂度分类器"列为未来工作。
7. **归因（attribution）与正确性（correctness）不是一回事**。RARR 能显著提升"每句话有证据支持"的比例，但不保证答案本身正确；下游任务精度甚至会小幅下降。

---

## 6. 评测基准与指标

**检索侧**：MS MARCO passage、TREC DL 2019/2020、BEIR（低资源零样本）、Mr.TyDi（多语言）。
指标：MRR@10、nDCG@10、Recall@{50,100,1k}。

**端到端 QA**：Natural Questions、TriviaQA、WebQuestions、SQuAD、AmbigNQ；多跳：HotpotQA、2WikiMultiHopQA、MuSiQue、IIRC。
指标：EM、F1、Accuracy。

**长文与忠实性**：ASQA、Biography、PubHealth、ARC-Challenge、QReCC、SQA。
指标：FactScore、citation precision/recall、Rouge、MAUVE、AttrAuto、Preservation。

**效率**：平均检索步数、端到端时延、token 消耗（Adaptive-RAG、HyPE、TreeHop 均报告）。

**注意**：不同论文的数据集与底座模型差异很大，跨论文数字不可直接横向比较；尤其是 "RAG" 基线在各文的定义不同。

---

## 7. 开放问题

1. **缺少直接度量 mismatch 的指标**。现有工作用下游 recall/EM 间接反映，无法定位"是词面、意图还是表示层的问题"。
2. **生成式扩展的忠实性**。伪文档/伪 query 可能引入事实错误，如何约束其"只补语义、不补事实"仍是开放问题。
3. **成本与延迟**。HyDE 类方法每次查询多一次生成；多跳方法多次检索。HyPE、TreeHop 是初步的降本尝试，但缺少系统性权衡研究。
4. **端到端联合优化**。rewriter、retriever、reader 目前多为分阶段优化；Rewrite-Retrieve-Read 的 RL 是早期尝试，训练不稳定、奖励稀疏。
5. **领域与多语言**。低资源领域（法律、医疗、工业）中，LLM 的假设文档质量下降，扩展反而有害；需要领域自适应的扩展策略。
6. **与 Agentic RAG 的结合**。把"改写/分解/纠错"变成 agent 的工具调用与规划步骤，是 2025–2026 的主要趋势，但评测与可控性仍不成熟。
7. **数据污染**。多跳与开放域基准被大规模预训练语料覆盖，生成式扩展可能"背答案"，导致收益被高估。

---

## 8. 论文清单（按 OpenAlex 引用数排序）

| # | 论文 | 年份 | 引用 | 类别 | 文件 |
|---|---|---|---|---|---|
| 1 | Least-to-Most Prompting | 2022 | 388 | 分解 | `markdown/2205.10625.md` |
| 2 | Precise Zero-Shot Dense Retrieval (HyDE) | 2022 | 371 | 假设文档 | `markdown/2212.10496.md` |
| 3 | Interleaving Retrieval with CoT (IRCoT) | 2022 | 359 | 迭代检索 | `markdown/2212.10509.md` |
| 4 | Measuring and Narrowing the Compositionality Gap (Self-Ask) | 2022 | 303 | 分解 | `markdown/2210.03350.md` |
| 5 | Query Rewriting for RAG (Rewrite-Retrieve-Read) | 2023 | 276 | 改写 | `markdown/2305.14283.md` |
| 6 | Adaptive-RAG | 2024 | 264 | 控制层 | `markdown/2403.14403.md` |
| 7 | Query2doc | 2023 | 220 | 假设文档 | `markdown/2303.07678.md` |
| 8 | Document Expansion by Query Prediction (doc2query) | 2019 | 215 | 索引侧 | `markdown/1904.08375.md` |
| 9 | Corrective RAG (CRAG) | 2024 | 187 | 检索后 | `markdown/2401.15884.md` |
| 10 | RARR | 2022 | 148 | 检索后 | `markdown/2210.08726.md` |
| 11 | Semantic Models for First-stage Retrieval (综述) | 2021 | 119 | 表示层 | `markdown/2103.04831.md` |
| 12 | Self-RAG | 2023 | 104 | 生成侧反思 | `markdown/2310.11511.md` |
| 13 | Generative Relevance Feedback (GRF) | 2023 | 49 | 生成式反馈 | `markdown/2304.13157.md` |
| 14 | Query Expansion by Prompting LLMs | 2023 | 26 | 扩展 | `markdown/2305.03653.md` |
| 15 | RaFe | 2024 | 21 | 改写（RL） | `markdown/2405.14431.md` |
| 16 | MILL | 2023 | 17 | 扩展（互验） | `markdown/2310.19056.md` |
| 17 | Best Practices of Query Expansion with LLMs | 2024 | 15 | 扩展 | `markdown/2401.06311.md` |
| 18 | Knowledge-Aware Query Expansion | 2024 | 5 | 扩展（关系） | `markdown/2410.13765.md` |
| 19 | Crafting the Path | 2024 | 4 | 改写（鲁棒） | `markdown/2407.12529.md` |
| 20 | HyPE | 2026 | 4 | 假设提示嵌入 | `markdown/2607.29402.md` |
| 21 | A Survey of Query Optimization in LLMs | 2024 | 2 | 综述 | `markdown/2412.17558.md` |
| 22 | DMQR-RAG | 2024 | 2 | 多查询改写 | `markdown/2411.13154.md` |
| 23 | TreeHop | 2025 | 0 | embedding 层改写 | `markdown/2504.20114.md` |

> Top10 的细致讲解见 [`top10.md`](top10.md)。
