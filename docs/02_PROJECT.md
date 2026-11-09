# 项目文档：Binding, not Matching

> 版本：2026-11-06（主线冻结：query-slot 因果定位与 CovBind 执行语义）
> 代码：`src/`　实验记录：`results/`　完整实验报告：`docs/03_EXPERIMENTS.md`

---

## 1. 一句话

RAG 的 query–证据语义 gap，主要不是"query 和证据不够像"，而是**query 里的需求变量没有被绑定**。绑定之后，第一段召回按跳次崩塌的现象消失；剩下的损失可以干净地分解为两个可度量、可分别优化的因子：

$$\text{gap 消除率}=\underbrace{\text{候选可达性}}_{\text{绑定/第一段}}\times\underbrace{\text{匹配能力}}_{\text{CE/蒸馏}}$$

经验值（MuSiQue，chunk 级，候选 top-20，175,642 个 chunk）：

$$\text{closure@3}=0.674\times0.734=0.495\quad(\text{LLM 教师口径})$$

---

## 2. 背景

### 2.1 标准 RAG 的形式

$$s(q,d)=\langle f(q),g(d)\rangle,\qquad \text{top-}k=\arg\max_d s(q,d)$$

生成阶段只看到这 k 个块。

### 2.2 语义 gap 的定义

**在语义上足以回答 query 的证据 $d^*$，其表示与 query 的相似度并不高**，因而缺席 top-k，或被噪声挤到后面。两类失败：

- **召回失败**：$d^*$ 根本没进候选；
- **排序失败**：进了候选，但排在噪声之后。

### 2.3 四类失配（`survey.md`）

| 类型 | 典型表现 |
|---|---|
| 词面/词汇 | 同义、缩写、跨语言 |
| 意图与粒度 | 问句 vs 陈述句；多跳 vs 单跳片段 |
| 表示/分布 | 短 query vs 长 chunk；口语 vs 书面 |
| 证据有效性 | 内容相关但过时/错误（不在本项目范围） |

本项目重新审视前三类，结论是：**在真实多跳语料上，主导失败不是前三类的"表示"问题，而是第一类里的一种特例——查询缺少中间实体（未绑定）。**

---

## 3. 问题形式化

### 3.1 需求与覆盖

- query 编译成需求集 $R=\{r_1,\dots,r_m\}$，每个 $r_i$ 是带类型的原子需求，可依赖前一个需求的答案。
- 证据单元 $d$ 对 $r_i$ 的 grounding 记作 $\text{ground}(r_i,d)\in\{0,1\}$。
- 证据集 $S$ 的覆盖：

$$\text{cov}(S)=\frac{1}{m}\sum_{i=1}^{m}\max_{d\in S}\text{ground}(r_i,d)$$

### 3.2 两因子分解

任何"只在候选集 $C$ 内选择"的方法满足：

$$\text{cov}(S)\le\underbrace{\frac{|\{i:\exists d\in C,\text{ground}(r_i,d)=1\}|}{m}}_{\text{可达性}}\times\underbrace{\text{选择效率}}_{\le 1}$$

- **可达性**由第一段检索与查询绑定决定；
- **选择效率**由匹配器与选择规则决定。

实测（MuSiQue，候选 top-50）：BM25 0.610 / MiniLM 0.760 / BGE 0.765 / HyDE 0.770；选择效率逐点 47–56%，覆盖式接近 100%。

### 3.3 为什么逐点 top-k 不够

逐点打分器独立给每个文档打分，**不知道其他文档已经覆盖了哪些需求**：

1. 无覆盖保证：构造 m 个高分干扰项即可让 top-k 覆盖为 0（$k<m$）。
2. 冗余盲：预算可能重复花在同一需求上。
3. 边际收益递减：覆盖率随 k 饱和（见 E1）。

覆盖最大化对集合单调次模，贪心有 $1-1/e$ 保证；因此**给定精确 grounding，选择问题已解决，难度在 grounding 的估计**。

---

## 4. 方法：CovBind（查询槽绑定 + 证据覆盖）

> 早期版本称 Plan-Conditioned Grounding（PAG）。单变量干预表明收益的稳定来源是**查询槽绑定**，因此方法名与组件命名收敛到 CovBind。CovBind 的贡献是执行语义与可达性诊断，不是新的匹配器训练目标。

### 4.1 流水线

```
query
  │
  ├─(1) Planner：query → 需求计划 [r1, r2, ...]，后一跳用 #1 引用前序答案
  │
  ├─(2) 逐跳：
  │      bind     把已得到的答案填入 #k
  │      retrieve BM25/dense 取候选
  │      ground   CE 给 (requirement, chunk) 打分
  │      read     读者从 top-3 证据中抽取答案
  │
  ├─(3) 查询槽替换：检索与重排的查询 = 绑定后的需求 r̂_i，不是原问题 q
  │
  ├─(3b) 覆盖选择：贪心最大化覆盖 − 冗余
  │
  └─(4) 停止：全部需求 grounded，或预算耗尽（可验证）
```

### 4.2 三个组件与证据

| 组件 | 作用 | 实验 | 结论 |
|---|---|---|---|
| **Planner** | query → 需求计划 | E4 | 自动规划与金标计划等价（+0.020，CI 跨 0） |
| **Binding** | 前序答案填入后续需求 | E2c / E3 | 召回 0.54–0.67 → 0.89–0.97；端到端 +0.090 EM |
| **Grounding 匹配器** | (requirement, chunk) → 满足概率 | E2b / E5 / E6 / E10 | 粗选用现成 CE 足够；高难 gap 蒸馏仅 +2.9 点 |
| **查询槽替换** | 检索/重排用 r̂_i 而非 q | E24 | 同计划同读者同重排器：EM 0.215 → 0.450（+0.235） |
| **覆盖选择** | 贪心最大化覆盖 | E1 / E2b / E23 | 三种实现差异 ≤0.02；选择规则不是杠杆 |

### 4.3 关键设计选择

1. **不做"失败后修复"**：对齐发生在检索时（用户明确否决 GASA/GRIP）。
2. **按需求绑定，而非自由生成子问题**：缺口直接渲染查询，停止条件可验证。
3. **覆盖信号叠加而非替换**：`claim_min`、硬阈值覆盖都显著更差。
4. **匹配器默认不训练**：E2b 显示现成 CE 达 oracle 的 90%；E6/E10 显示高难 gap 上蒸馏有正收益但很小。

---

## 5. 四个贡献

### C1 诊断：gap 是绑定问题，不是表示问题

把"query–证据不匹配"从**表示对齐**重述为**变量绑定 + 集合覆盖**，并给出直接度量（不是用 EM 间接反映）。

| 证据 | 数值 |
|---|---|
| 按跳的召回崩塌（未绑定） | 0.915 → 0.615 → 0.673 → **0.538** |
| 绑定后（同语料、同标签，仅改查询） | 0.970 / 0.935 / 0.921 / **0.923** |
| chunk 级 gap 占比（dense） | **17.6%**（95/540 跳；后扩到 783/1916） |
| 损失分解 | 0.610 × 0.47 = 0.29 |
| **查询槽反事实（E24）** | 同计划/读者/重排器/预算：EM **0.215 → 0.450**（+0.235 [+0.170,+0.300]） |
| 绑定→覆盖（E23，2-hop n=56） | 覆盖率 0.598 → 0.866；全证据率 32.1% → 75.0%；EM 0.286 → 0.554 |
| 训练分布对照（E23 2×2） | 需求重排器 vs 问题重排器（同数据同负例）：−0.018 [−0.041,+0.002] |

### C2 方法：CovBind

| 证据 | 数值 |
|---|---|
| 端到端（MuSiQue 200，域内 CE） | ours **0.450** vs IRCoT 0.315、Vanilla 0.195 |
| 端到端（HotpotQA，200 题） | ours 0.675 vs IRCoT 0.660（Δ+0.015，打平） |
| 2×2 消融 | 绑定 +0.14，覆盖选择 +0.08（旧口径）；E23 显示选择规则只差 ≤0.02 |
| 自动规划 | 与金标计划等价（+0.020，CI 跨 0） |

### C3 负结果：两条流行假设不成立

1. **查询侧改写不是干 gap 的正确杠杆**：同义改写只消除 **9.5%** 的 gap case，陈述句投影 31.6%，CE 语义重排 35.8%（E5）。
2. **LLM 的绝对充分性判定不是上界**：在 582 个真实 gap case 上，LLM 的 top-3 为 **0.170**，低于 base CE 的 **0.198**；成对判定中 **47.9% 是平局**（E10/E11）。
3. 推论：**蒸馏的收益被教师上限锁死**。真实 top-20 口径下蒸馏只从 0.198 提到 0.227（+2.9），远低于精选候选集上的 +9.8。

---

## 6. 与已有工作的关系

| 工作 | 它在优化什么 | 为什么不够 |
|---|---|---|
| **HyDE / Query2Doc** | 让 query 的**措辞**更像文档 | E5：措辞换了，gap 只消 9.5% |
| **SETR (ACL'25)** | 在候选集内按信息需求**选集合** | 候选里没有的证据选不出来（天花板 0.61–0.77）；靠提示、非第一段 |
| **SURE-RAG (2026)** | 证据充分性用于**弃答** | 验证/弃答，不改 retrieve 什么 |
| **PureCover** | 覆盖式 **rerank** | 同上，候选集内 |
| **ReSCORE (ACL'25)** | 用 LLM 概率**训 dense retriever** | 标签是逐文档相关性，不是逐需求 grounding |
| **IRCoT / Self-Ask** | **迭代**检索 / 子问题 | 每步仍是整句 top-k；起作用的是绑定，不是迭代本身 |
| **Adaptive-RAG** | 按复杂度路由 | 正交，可作为控制层 |

**空白**：把 gap 诊断为绑定问题，并给出"可达性 × 匹配能力"的直接度量与对应方法。

---

## 7. 数据

### 7.1 MuSiQue（主数据集）

- `dgslibisey/MuSiQue` validation，seed=42。
- 每题 19–20 段；`question_decomposition` 给出每跳子问题、答案、**对应段落索引**。
- 用途：逐跳召回、绑定、覆盖、端到端、gap 测量。
- 规模：实验 E1–E6 用 200 题；E8–E11 扩到 1000 / 2417 题。

### 7.2 HotpotQA（第二数据集）

- `hotpot_qa` distractor validation，seed=42，200 题；每题 10 段。
- 只有段落级金标，无逐跳分解；用于端到端复验。

### 7.3 Chunk 级语料（E5/E6/E10）

- 40-token 滑窗、stride 20。
- gold chunk = 含该跳答案 token 序列的窗口（词边界匹配）。
- 200 题 → 14,289 chunk；2417 题 → 175,642 chunk。

---

## 8. 全部参数

### 8.1 检索

| 参数 | 值 |
|---|---|
| 分词 | `[a-z0-9]+`，小写 |
| BM25 | `k1=1.5`, `b=0.75` |
| 候选数 | 10（pilot）/ 50（E1/E2b）/ 20（E5/E6/E10 评测） |
| Chunk | 40 token，stride 20 |
| Dense | `all-MiniLM-L6-v2`（384d，mean pooling） |
| 对照 | `BAAI/bge-small-en-v1.5`、HyDE |

### 8.2 服务

| 用途 | 端点 |
|---|---|
| 生成/判定 | `10.94.199.216/217:8000/v1`，`qwen3.6-27b`，`temperature=0`，`enable_thinking=False` |
| CE / Embedding | `10.52.100.210:8833`（`/rerank`、`/score`、`/embed`，原始 logits） |
| 文件分发 | `10.25.71.43:8869` |

### 8.3 方法

| 参数 | 值 |
|---|---|
| claim 数 | 1–4（平均 1.95） |
| 覆盖阈值 τ | 0.3 / 0.5 / 0.7 |
| 读者证据数 | 逐跳 top-3；最终 top-8 |
| 读者 `max_tokens` | 32 |

### 8.4 蒸馏（E7–E11）

| 参数 | 值 |
|---|---|
| 学生 | `ms-marco-MiniLM-L-6-v2`（22M） |
| 逐点数据 | 5.4k → 4.0k → 16.6k → **36.9k**（E7→E10） |
| 成对数据 | **4,450**（E11，来自 8,546 条、47.9% 平局被丢弃） |
| 损失 | `BCE + 0.5·relu(1.0 − (s_gold − s_neg))` |
| epochs / lr / batch / max_len | 4 / 2e-5 / 64 / 128 |
| 切分 | 按 qid 70/30，seed=0 |

---

## 9. 指标

| 指标 | 定义 |
|---|---|
| Recall@k | 金标进入前 k 的比例 |
| 覆盖率 | 被满足需求数 / 总需求数 |
| 候选 recall@k | 可达性 |
| **gap 占比** | gold 在 top-50 内、top-3 外的比例 |
| **closure@3 / @10** | gap case 中被排进前 3 / 前 10 的比例 |
| EM / F1 | SQuAD 归一化 |
| AUC | 打分器区分正负 |

区间：按题 1000 次 bootstrap；方法对比为配对 bootstrap。

---

## 10. 目录与运行

```
docs/        01_PROCESS.md / 02_PROJECT.md / 03_EXPERIMENTS.md
src/         llm, ce_client, data, musique, retriever, claims, coverage,
             slotjudge, eval, probe, e0–e11, run_pilot
results/     全部实验原始结果
data/raw/    musique200/1000/2417, hotpot20/200
deploy/      本地 CE 服务
dist/        E7–E11 GPU 训练包
```

```bash
python3 -m src.run_pilot --stage data --limit 20
python3 -m src.musique
python3 -m src.e1 --n 200
python3 -m src.e2b_remote --n 200
python3 -m src.e2c_hopquery
python3 -m src.e3_binding --n 200
python3 -m src.e4_pipeline --n 200 --plan auto
python3 -m src.e5_gap --n 200 --stage all
python3 -m src.e6_llm_upper --n 200
python3 -m src.e8_mine --n 2417 --data musique2417.jsonl
python3 -m src.e8_label
python3 -m src.e10_compare
```

---

## 11. 已知限制

1. **端到端增益有限**：MuSiQue +0.075 EM，HotpotQA 打平；绑定单独使用只追平 IRCoT。
2. **匹配器收益小**：真实 top-20 口径下蒸馏 +2.9 点；LLM 教师不是上界。
3. **两个英文 Wikipedia 数据集**；2Wiki 镜像不可用。
4. **E2c 用金标绑定**；规划误差只在 E4 间接体现。
5. **理论是草图**：没有正式命题，只有经验曲线与分解式。
6. **标签噪声**：答案字符串匹配与教师判定的一致率约 70%。
