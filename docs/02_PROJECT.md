# 项目文档：Query–Evidence 语义 gap 的测量与消除

> 版本：2026-11-04
> 代码：`src/`　实验记录：`results/`　实验协议：`docs/03_EXPERIMENTS.md`

---

## 1. 一句话

RAG 检索失败的主要来源不是编码器不够强，而是两件可以被分别度量的事：**证据有没有进入候选（可达性）**，以及**进入候选后能不能被排到前面（匹配能力）**。本项目把"语义 gap"拆成这两个乘性因子，分别给出测量与消除手段。

$$\text{gap 消除率} = \underbrace{\text{候选可达性}}_{\text{绑定/第一段}} \times \underbrace{\text{匹配器能力}}_{\text{CE/蒸馏}}$$

---

## 2. 背景

### 2.1 RAG 的标准形式

标准 RAG 用同一个相似度函数比较 query 与文档块：

$$s(q,d) = \langle f(q), g(d) \rangle,\qquad \text{top-}k = \arg\max_d s(q,d)$$

生成阶段只看到这 k 个文档块。

### 2.2 什么是语义 gap

**在语义上足以回答 query 的证据 $d^*$，其表示与 query 的相似度并不高**，因此缺席 top-k，或被噪声挤到后面。表现为两类失败：

- **召回失败**：正确证据根本没进 top-k；
- **排序失败**：进了候选，但排在噪声之后。

### 2.3 四类失配（来自 `survey.md`）

| 类型 | 典型表现 | 例子 |
|---|---|---|
| 词面/词汇 | 同义、缩写、跨语言 | 问"心肌梗死"，文档写"AMI" |
| 意图与粒度 | 问句 vs 陈述句；多跳 vs 单跳片段 | "A 的导师的母校" |
| 表示/分布 | 短 query vs 长 chunk；口语 vs 书面 | "那它呢？" |
| 证据有效性 | 内容相关但过时/错误/实体不一致 | 问 2026 年数据，检索到 2021 年 |

本项目主要处理前三类；第四类属于检索后校验，不在范围内。

---

## 3. 问题定义与形式化

### 3.1 需求与覆盖

- 查询被编译成需求集 $R=\{r_1,\dots,r_m\}$，每个 $r_i$ 是一个带类型的原子需求（可能依赖前一个需求的答案）。
- 语料中的证据单元 $d$ 对 $r_i$ 的 grounding 记为 $\text{ground}(r_i,d)\in\{0,1\}$。
- 证据集 $S$ 的覆盖：

$$\text{cov}(S)=\frac{1}{m}\sum_{i=1}^{m}\max_{d\in S}\text{ground}(r_i,d)$$

### 3.2 两个乘性因子

对任何"只在候选集 $C$ 内选择"的方法：

$$\text{cov}(S)\le \underbrace{\frac{|\{i:\exists d\in C,\ \text{ground}(r_i,d)=1\}|}{m}}_{\text{候选可达性}}\times\underbrace{\text{选择效率}}_{\le 1}$$

- **可达性**由第一段检索决定：证据是否进入候选。
- **选择效率**由匹配器与选择规则决定：进入候选后能否被选中。

经验值（MuSiQue，200 题，候选 top-50）：

- 可达性：BM25 0.610 / MiniLM 0.760 / BGE 0.765 / HyDE 0.770
- 选择效率：逐点 47%–56% / 覆盖式选择接近 100%

### 3.3 为什么逐点 top-k 不够

逐点打分器对每个文档独立打分，**不知道其他文档已经覆盖了哪些需求**，因此：

1. 没有覆盖保证：构造 m 个高分干扰项即可让 top-k 覆盖为 0（$k<m$）。
2. 冗余盲：预算可能花在同一个需求上。
3. 加 k 的边际收益递减（覆盖率随 k 饱和，见 E1）。

覆盖最大化对集合是单调次模的，贪心有 $1-1/e$ 保证；因此**给定精确 grounding，选择问题已解决，全部难度在 grounding 的估计**。

---

## 4. 方法：PAG（Plan-Aware Grounding）

### 4.1 流水线

```
query
  │
  ├─(1) Planner：query → 需求计划 [r1, r2, ...]，后一跳可用 #1 引用前一跳答案
  │
  ├─(2) 逐跳：
  │      bind：把已得到的答案填入 #k
  │      retrieve：BM25/dense 取候选（top-50）
  │      ground：CE 给 (requirement, chunk) 打分
  │      read：读者从 top-3 证据中抽取答案
  │
  ├─(3) 覆盖选择：若一次面对多个需求，贪心最大化覆盖 − 冗余
  │
  └─(4) 停止：全部需求 grounded，或预算耗尽（可验证，不靠模型说停）
```

### 4.2 三个组件与证据

| 组件 | 作用 | 对应实验 | 结论 |
|---|---|---|---|
| **Planner** | query → 需求计划 | E4 | 自动规划与金标计划等价（+0.020，CI 跨 0） |
| **Binding** | 把前序答案填入后续需求 | E2c / E3 | 召回 0.54–0.67 → 0.89–0.97；端到端 +0.090 EM |
| **Grounding 匹配器** | (requirement, chunk) → 满足概率 | E2b / E5 / E6 | 现成 CE 够用于粗选；高难度 gap 需更强匹配器 |
| **覆盖选择** | 贪心最大化覆盖 | E1 / E2b | 选择效率 47% → 90% of oracle |

### 4.3 关键设计选择（以及为什么）

1. **不做"失败后修复"**：对齐发生在检索时，不是生成后（用户明确否决 GASA/GRIP）。
2. **匹配器不训练**（粗粒度）：E2b 显示现成 CE 达 oracle 的 90%，训练收益撑不起独立贡献。
3. **高难度 gap 需要训练**：E6 显示可达子集上 CE 56% vs LLM 73%，17 点可学 → E7 蒸馏。
4. **按需求绑定，而不是自由生成子问题**：缺口直接渲染查询，停止条件可验证。
5. **覆盖信号叠加而非替换**：`claim_min`、硬阈值覆盖都显著更差。

---

## 5. 与已有工作的关系

| 工作 | 做了什么 | 本项目差别 |
|---|---|---|
| **HyDE / Query2Doc** | query → 伪文档，仍靠相似度 | 我们改的是目标（覆盖）与绑定；E5 显示纯改写只消 9.5% 的 gap |
| **SETR (ACL'25)** | CoT 识别信息需求，按集合完整性选段 | 它在 rerank 候选集内、靠提示；我们做第一段可达性 + 可度量的 gap |
| **SURE-RAG (2026)** | 证据充分性作为 set-level 性质，用于弃答 | 它是验证/弃答，不改检索 |
| **PureCover** | 覆盖式 rerank | 同上，候选集内 |
| **ReSCORE (ACL'25)** | LLM 概率训练 dense retriever | 标签是逐文档相关性；我们按需求 grounding |
| **IRCoT / Self-Ask** | 迭代检索 / 子问题 | 我们用绑定 + 需求级选证；E4 中 MuSiQue 上 +0.075 EM，HotpotQA 打平 |
| **Adaptive-RAG** | 按复杂度路由 | 正交，可作为控制层 |

**空白**：按需求训练/执行的第一段 grounding；把"逐点 top-k 的失败根因"量化成可达性 × 匹配能力。

---

## 6. 数据

### 6.1 MuSiQue（主数据集）

- 来源：`dgslibisey/MuSiQue` validation，seed=42 取前 200 题。
- 每题 19–20 段；`question_decomposition` 给出每一跳的子问题、答案、**对应段落索引**。
- 这是唯一同时提供"逐跳标签 + 段落池"的公开数据，因此作为主实验。
- 缓存：`data/raw/musique200.jsonl`。字段：`qid/question/answer/n_hops/hops/docs`。

### 6.2 HotpotQA（第二数据集）

- `hotpot_qa` distractor validation，seed=42，200 题。
- 每题 10 段；`supporting_facts` 给段落级金标，**没有逐跳分解**。
- 用途：端到端复验（E4）；不做逐跳召回分析。
- 缓存：`data/raw/hotpot200.jsonl`。

### 6.3 Chunk 级语料（E5/E6）

- 段落切成 **40-token 滑窗、stride 20**。
- gold chunk = 含该跳答案 token 序列的窗口（词边界匹配）。
- 200 题 → 14,289 个 chunk，540 跳。

---

## 7. 全部参数

### 7.1 检索

| 参数 | 值 | 说明 |
|---|---|---|
| 分词 | `[a-z0-9]+`，小写 | 句级与文档级一致 |
| BM25 | `rank_bm25.BM25Okapi`, `k1=1.5`, `b=0.75` | 与交接文档一致 |
| 候选数 | pilot 10；E1/E2b 50；E5 全库；E4 50 | |
| Chunk | 40 token，stride 20 | 避免分句器在 `c.` 处截断 |
| Dense | `sentence-transformers/all-MiniLM-L6-v2`（384d，mean pooling） | |
| 强检索器对照 | `BAAI/bge-small-en-v1.5` | |
| HyDE | 每题生成一段 2–3 句伪文档，再用 MiniLM 编码 | |

### 7.2 模型服务

| 用途 | 端点 | 备注 |
|---|---|---|
| 生成/判定 | `http://10.94.199.216:8000/v1`、`...217:8000/v1` | `qwen3.6-27b`，`temperature=0`，`enable_thinking=False`，双端点轮询 |
| CE / Embedding | `http://10.52.100.210:8833` | `/rerank`（`documents`，返回 logits）、`/score`、`/embed`；GPU |
| CE 模型 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 服务端 `/models/ce-msmarco` |
| Embedding 模型 | `all-MiniLM-L6-v2` | 服务端 `/models/minilm` |

### 7.3 方法超参

| 参数 | 值 |
|---|---|
| claim 数 | 1–4（平均 1.95） |
| 覆盖阈值 τ | {0.3, 0.5, 0.7}（结果对 τ 不敏感） |
| 读者证据数 | 逐跳 top-3；最终答案 top-8 |
| 读者 `max_tokens` | 32 |
| 规划 `max_tokens` | 256 |
| LLM 缓存 | sqlite，key = sha1(model + messages + params) |
| 并发 | 8 线程（LLM） |

### 7.4 E7 蒸馏超参

| 参数 | 值 |
|---|---|
| 初始化 | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 训练对 | 5400（540 跳 × dense top-10） |
| 目标 | LLM 分数 / 10，BCEWithLogits |
| epochs / lr / batch / max_len | 3 / 2e-5 / 64 / 128 |
| 切分 | 按问题 id 分组，70/30，seed=0 |
| 失败线 | 可达子集 closure@3 < 0.62 |

---

## 8. 指标定义

| 指标 | 定义 | 用在哪 |
|---|---|---|
| MRR@k | 第一个金标排名的倒数 | pilot |
| nDCG@k | 二值相关性，IDCG 取前 \|G\| 位 | pilot |
| Recall@k | 金标被前 k 覆盖的比例 | E1–E4 |
| 覆盖覆盖率 | 被满足的需求数 / 总需求数（逐跳金标） | E1/E2b |
| 候选 recall@k | 金标进入前 k 的比例 = 可达性 | E2a/E5 |
| **gap 占比** | gold 在 top-50 内但 top-3 外的比例 | E5 |
| **closure@3** | gap case 中被重新排进 top-3 的比例 | E5/E6/E7 |
| EM / F1 | SQuAD 归一化（小写、去标点、去冠词） | E3/E4 |
| AUC | 打分器区分正负样本的能力 | pilot/E2b |

置信区间：按题做 1000 次 bootstrap 取 2.5%/97.5% 分位；方法对比用**配对** bootstrap。

---

## 9. 目录结构

```
query-evidance/
├── docs/                  # 本目录的三份文档
├── src/
│   ├── llm.py             # 端点轮询 / 缓存 / 关思考 / JSON 修复
│   ├── ce_client.py       # 远程 CE/embedding 客户端（logits → sigmoid）
│   ├── data.py            # HotpotQA 加载与池化
│   ├── musique.py         # MuSiQue 加载（逐跳标签）
│   ├── retriever.py       # 文档级/句级 BM25
│   ├── claims.py          # query → 原子 claim
│   ├── coverage.py        # 聚合方式（max/mean/min/阈值/hybrid）
│   ├── slotjudge.py       # LLM 槽位覆盖判定（B 线上界）
│   ├── eval.py            # 指标与 bootstrap
│   ├── probe.py           # E0-a 等价改写探针
│   ├── e0.py              # E0-b 冗余盲探针
│   ├── e1.py / e1_dense.py / e1_analysis.py / e2a_hyde.py
│   ├── e2b_ce.py / e2b_remote.py / e2c_hopquery.py
│   ├── e3_binding.py / e4_pipeline.py
│   ├── e5_gap.py / e6_llm_upper.py / e7_distill.py
│   └── run_pilot.py
├── results/               # 全部实验的 jsonl/json/md
├── data/raw/              # musique200 / hotpot200 / hotpot20
├── data/index/            # embedding 缓存（未入 git）
├── deploy/ce_server.py    # 本地 CE 服务（可选）
├── dist/e7/               # GPU 训练包
├── cache/                 # LLM 响应缓存（未入 git）
├── RESEARCH_PLAN.md / PILOT.md / HANDOVER.md / survey.md / top10.md
└── docs/01_PROCESS.md / 02_PROJECT.md / 03_EXPERIMENTS.md
```

---

## 10. 运行

```bash
cd /disk1/query-evidance
# 外网（下载数据）才需要代理；内网 LLM/CE 必须绕过代理
export http_proxy="<proxy>"; export https_proxy="$http_proxy"

python3 -m src.run_pilot --stage data --limit 20
python3 -m src.musique                      # 下载并缓存 MuSiQue 200
python3 -m src.e1 --n 200                   # 覆盖率与损失分解
python3 -m src.e1_dense --n 200 --model BAAI/bge-small-en-v1.5
python3 -m src.e2b_remote --n 200 --cand 50 # CE 覆盖选择
python3 -m src.e2c_hopquery                 # 绑定因果实验
python3 -m src.e3_binding --n 200           # 误差传播
python3 -m src.e4_pipeline --n 200 --plan auto --dataset musique
python3 -m src.e5_gap --n 200 --stage all   # 语义 gap 测量与消除
python3 -m src.e6_llm_upper --n 200 --topn 20
python3 -m src.e7_distill --n 200 --stage collect   # 蒸馏数据
```

所有长任务都按题增量写盘（`results/*.jsonl`），可断点续跑；LLM 响应按 sha1 缓存。

---

## 11. 术语

| 术语 | 含义 |
|---|---|
| gap case | gold 在候选 top-50 内但 top-3 外的样本 |
| 可达性 | gold 是否进入候选集 |
| 匹配能力 | 进入候选后能否被排到前面 |
| 绑定 | 把前序跳的答案填入后续跳的查询 |
| 覆盖选择 | 按需求覆盖而非逐点分数选证据集 |
| closure@3 | gap case 被重新排进 top-3 的比例 |

---

## 12. 已知限制

1. 只有英文 Wikipedia 两个数据集；2Wiki 镜像不可用。
2. 读者（top-3/top-8）是端到端上限；oracle 绑定下跳级 EM 仅 0.39–0.52。
3. E7 蒸馏结果未回传。
4. 理论部分只有经验曲线与草稿（`RESEARCH_PLAN.md` §8），没有正式命题。
5. LLM 判定可能借用参数知识；E2c 的绑定用的是金标答案，规划误差未计入 E2c。
