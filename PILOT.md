# Pilot 结果：claim 覆盖式聚合 vs 相似度检索

> 日期：2026-11-04　｜　状态：20/20 题跑完　｜　代码：`src/`　｜　原始结果：`results/`

## 1. 结论

1. **前提成立**：由 LLM 直接判定槽位覆盖（B 线，oracle 上界）把 MRR@10 从 0.790 提到 **0.957**（用 BM25 打破平局），配对 bootstrap 差值 **+0.167 [+0.058, +0.283]**，CI 明确不含 0。语义 gap 确实不是相似度能表达的。
2. **瓶颈在匹配函数，不在分解**：用 BM25 给 claim 打分（A 线，零额外调用）只有 0.754，单独用还不如整句 BM25；最好的廉价组合 `hybrid_0.3` 到 0.846，只回收了 oracle 增益的约 1/3（+0.055 vs +0.167）。下一步该做的是**把判定能力蒸馏到可批量计算的匹配器**，而不是继续调 prompt。
3. **"覆盖"必须是软的**：严格全覆盖（`claim_min`）比 BM25 低 **0.323** MRR（CI 不含 0），硬阈值计数（`claim_cover`）低 0.242。噪声下取 min 等于被最差的一跳支配。
4. **覆盖信号应当叠加而非替换**：`hybrid_0.3`（0.7×BM25 + 0.3×claim 均值）是唯一方向性优于 BM25 的廉价方案（+0.055 MRR，+0.05 Recall@1），但 20 题上 CI 仍跨 0。
5. **claim 信号对"这段是否含答案"更敏感**：`claim_max` 的 AUC(answer) 0.707 vs BM25 0.609，而 MRR 不升——说明它抓到的是"证据充分性"，不是"主题相关性"，两者确实可分离。

## 2. 设置

- 数据：HotpotQA `distractor` validation，seed=42 取 20 题，每题 10 段候选（标准 distractor 设定）。
- 检索：`rank_bm25`，`k1=1.5, b=0.75`；文档级与句级两个索引。
- 模型：`qwen3.6-27b`，`temperature=0`，`enable_thinking=False`，双端点轮询。
- 成本：243 次唯一 LLM 响应（20 次 claim 投影 + 200 次槽位判定 + 修复/冒烟），全部命中 sqlite 缓存后零调用。

## 3. 方法

- **claim 投影**：`query → 1–4 条原子声明`，中间跳是可核验的完整事实，最后一跳带 `?` 占位符；`$cX` 只允许引用更早的 claim。平均 **1.95** 条/题（17/20 为 2 跳）。
- **A 线打分**：claim 对句级 BM25 打分 → 每篇取句级 max → 每条 claim 在 10 篇上 min-max 归一 → 三种聚合（max / mean / min / 阈值覆盖）。
- **B 线（oracle）**：每 `(query, passage)` 一次 LLM 调用，逐槽位输出 `covered + 逐字 span`，分数 = 覆盖比例。span 逐字校验通过率 **1.000**（200/200）。

## 4. 结果（20 题，均值 [95% CI]）

| 方法 | MRR@10 | nDCG@10 | Recall@1 | Recall@2 | AUC(answer) | ΔMRR vs BM25 |
|---|---|---|---|---|---|---|
| bm25_whole | 0.790 | 0.799 | 0.325 | 0.575 | 0.609 | — |
| claim_max | 0.754 | 0.791 | 0.275 | 0.650 | **0.707** | -0.036 [-0.245,+0.179] |
| claim_mean | 0.668 | 0.730 | 0.250 | 0.450 | 0.648 | -0.122 [-0.344,+0.120] |
| claim_min | 0.467 | 0.573 | 0.150 | 0.250 | 0.457 | **-0.323 [-0.562,-0.068]** |
| claim_cover (τ=0.5) | 0.548 | 0.665 | 0.175 | 0.375 | 0.619 | -0.242 [-0.445,-0.035] |
| claim_cover_tb | 0.660 | 0.726 | 0.250 | 0.425 | 0.648 | -0.130 [-0.351,+0.118] |
| hybrid_0.3 | **0.846** | **0.818** | **0.375** | 0.550 | 0.621 | +0.055 [-0.011,+0.147] |
| hybrid_0.5 | 0.850 | 0.815 | 0.375 | 0.550 | 0.610 | +0.060 [-0.049,+0.171] |
| judge_oracle（无平局规则） | 0.938 | 0.870 | 0.450 | 0.675 | 0.641 | +0.147 [+0.001,+0.304] |
| **judge_tb (B，BM25 破平)** | **0.957** | **0.908** | **0.475** | 0.675 | 0.653 | **+0.167 [+0.058,+0.283]** |

## 5. 案例

> 注意：判定器在 2/20 题上对全部 10 篇都给出 0 分，此时排序完全由平局规则决定。下表案例均已用 **BM25 破平**复算，排除了这一伪影。

**主题相关但不含所需槽位（判定器胜出）**

> *What nationality was Oliver Reed's character in the film Royal Flash?*　gold: `Otto von Bismarck`, `Royal Flash (film)`
> - BM25 top-1：`Ivan Dragomiloff`（角色条目，主题相关但无国籍），gold 第 2
> - judge_tb top-1：`Royal Flash (film)`，gold 第 1

**高相似、零覆盖（判定器胜出）**

> *The actor that played Gutthi on Comedy nights with Kapil Show also starred in which film?*　gold: `Sunil Grover`, `Vaisakhi List`
> - BM25 top-1：`Bhaji in Problem`（同为该演员作品，词面重合），gold 第 3
> - judge_tb top-1：`Sunil Grover`，gold 第 1

**关系同义词（判定器与 hybrid 都胜出）**

> *If the Charhki Dadri crash was less dangerous than the Tenerife airport disaster, ...*
> - BM25 top-1：`1938 Jersey Airport disaster`（"crash"/"disaster" 词面重合），gold 第 3
> - hybrid_0.3 / judge_tb top-1：`Charkhi Dadri mid-air collision`，gold 第 1

**廉价近似只修一半**

> *In what county was Duffy Jackson born?*　gold: `Duffy Jackson`, `Freeport, New York`
> - BM25 top-1：`Keith Duffy (bassist)`（"Duffy" 词面主导），gold 第 7
> - hybrid_0.3：gold 第 4（有改善但没进 top-1）
> - judge_tb：该题判定器全 0，退回 BM25 排序，**没有修复**

**判定器漏判**

> *What is the first two words of the fifth studio album of Joseph Edgar Foreman?*
> - 判定器对全部 10 篇给 0 分，说明它对"取字符串前两个词"这类非事实型需求无能为力；这类题不应计入覆盖式方法的收益。

## 6. 已知问题

1. **判定器经常全 0，平局规则会伪造收益**。200 次判定里只有 39 个 `covered=true`，2/20 题上是全 0。第一版报告用"文档原始顺序"破平，把这类题的排序当成了判定器的功劳（Duffy Jackson、Biathlon 两个案例都是伪影）。现统一改用 BM25 破平（`judge_tb`），并以此为准。
2. **judge 本质是逐对相关性判定器**，不是"覆盖度"的精细度量：覆盖率被压成 0/1，槽位权重、部分覆盖都没有体现。它给出的是上界，不是方法。
3. **oracle 可能借用参数知识**。判定器看到问题与段落，理论上可以"凭记忆"判覆盖。span 逐字校验通过只说明它抄了原文，不排除泄漏。需要跑一个**段落置换对照**（把 question 换成同实体的另一个问题，看覆盖判定是否随之崩塌）来验证。
4. **distractor 只有 10 篇候选，BM25 接近天花板**。这里的绝对数值不可外推；正式实验必须至少在 2Wiki/MuSiQue 自带池与全量 Wikipedia 上复验。
5. **claim 打分用的是句级 BM25**，与整句 BM25 不同源，两个变量（分解 vs 匹配）没有完全解耦。
6. **20 题样本太小**：`hybrid_0.3` 的 CI 仍跨 0，只能当方向性证据。

## 7. 下一步（按优先级）

1. **匹配器蒸馏**：用 B 线的 200 条判定做监督，训练/微调一个小 cross-encoder（CPU 可推理），目标是回收 oracle 增益的 2/3 以上；同时把每 query 成本压到 1 次 LLM + 可批量编码。
2. **扩数据**：2Wiki + MuSiQue 各 200 题，自带上下文池；同时加 60 篇跨题池的难负例，避免 10 候选的饱和。
3. **段落置换对照**，排除参数知识泄漏。
4. 若匹配器成立，再实现**缺口驱动下一跳**（用未覆盖 claim 渲染查询）与可验证停止条件。

## 8. 复现

```bash
cd /disk1/query-evidance
export http_proxy="<PROXY>" && export https_proxy="$http_proxy"   # 仅数据下载需要

python3 -m src.run_pilot --stage data --limit 20    # HotpotQA -> data/raw/hotpot20.jsonl
python3 -m src.run_pilot --stage claims --workers 8 # 1 LLM call/question
python3 -m src.run_pilot --stage judge  --workers 8 # 1 LLM call/(question, passage)
python3 -m src.run_pilot --stage report             # results/pilot_report.md
```

内网端点必须绕过代理（`llm.py` 里已写死 `proxies={}`）。所有结果按题增量写入 `results/*.jsonl`，可断点续跑。
