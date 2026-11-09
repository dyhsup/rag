# 绑定而非匹配：多跳检索中的查询槽因果性

**Binding, Not Matching: Query-Slot Causality in Multi-Hop Retrieval**

ACL 主会中文论文初稿

---

## 摘要

检索增强生成（RAG）通常把查询视为一段文本，并通过相似度或 cross-encoder 对候选证据排序。该范式在多跳问题上存在一个被忽视的限制：后续需求中的中间实体往往不是原问题中的显式词项，因此证据并非“表达得不够像”，而是没有被要求去回答正确的查询对象。本文提出一个因果诊断视角：**多跳 query–evidence gap 的主要发生位置是查询槽，而非匹配器本身。**

我们将多跳查询表示为带变量和依赖边的需求程序，提出 CovBind 执行流程：编译需求、绑定前序答案、用绑定后的需求替换检索查询槽、按需求选择证据，并以可验证覆盖作为停止条件。我们的核心实验固定语料、金标标签、需求计划、读者、重排器和每跳证据预算，只改变查询槽：整句问题对比绑定后的需求。在 MuSiQue 400 题上，EM 从 0.170 提升到 0.425，配对增益为 +0.255，95% bootstrap 区间为 [+0.210,+0.303]。按跳的 dense Recall@50 从 0.915/0.615/0.673/0.538 恢复到 0.970/0.935/0.921/0.923。

我们进一步把覆盖损失分解为候选可达性与选择效率。BM25、MiniLM、BGE-small 和 HyDE+MiniLM 的候选可达性分别为 0.610、0.760、0.765 和 0.770，而逐点选择效率长期停留在 0.47–0.56。证据完整时 reader EM 约为 0.62–0.65，证据不完整时仅约 0.16，说明覆盖是主要瓶颈。跨数据集实验显示，方法在 MuSiQue 上最稳定，在 2Wiki 与 HotpotQA 上基本持平，在 MultiHopRAG 新闻语料上获得边缘收益。最后，我们系统报告查询改写、答案似然、LLM 充分性判断、绑定质量控制和多种匹配侧机制的负结果。本文的贡献不是提出新的 matcher，而是通过可复现的单变量干预证明：在研究设置中，**先绑定查询对象比继续强化相似度排序更关键。**

**关键词：** 检索增强生成；多跳问答；查询槽；证据可达性；需求覆盖；因果诊断

---

## 1 引言

RAG 的标准检索过程可以写成：给定查询 $q$ 和语料 $\\mathcal{C}$，为每个证据单元 $d$ 计算 $s(q,d)$，再选择 top-$k$。当检索失败时，最常见的修补是训练更强的 dense retriever 或 cross-encoder。它们提高了 query 与 evidence 的表示或排序质量，但通常保持查询对象不变。

这一点在多跳问题中十分关键。考虑问题：

> Who is the spouse of the actor of Ethan in *A Dog's Purpose*?

第一跳需要找出演员 Dennis Quaid；第二跳真正需要检索的是“Dennis Quaid spouse”。原问题本身没有包含 Dennis Quaid。若第二跳继续使用原问题，即使匹配器足够强，查询也没有指向正确证据的关键变量。

本文研究以下问题：

> 多跳 query–evidence gap 的主因究竟是表示匹配能力不足，还是查询中缺少已知的中间变量？

我们不通过比较一系列不同模型来回答，而是采用单变量干预：在相同计划、相同语料、相同 reader、相同 reranker 和相同预算下，只替换检索与重排使用的 query slot。

本文的主要发现如下：

1. **查询槽是可观测的因果杠杆。** MuSiQue 400 题上，只将整句问题替换为绑定需求，EM 提升 +0.255。
2. **绑定首先改善候选可达性。** 后续 hop 的 Recall@50 从 0.538–0.673 恢复至 0.921–0.935。
3. **覆盖而非单段相关性是端到端瓶颈。** 全部需求证据同时出现时 EM 约 0.62–0.65；证据不完整时约 0.16。
4. **方法有结构性边界。** 增益需要绑定缺口较大且预算覆盖所有需求；2Wiki/HotpotQA 的 null 结果并非简单随机波动。
5. **更强 matcher 不是本文贡献。** 训练 CE 仅作为工程组件；Role-OT 和结构事实 grounding 作为探索性实现未通过初筛，不纳入主方法主张；追加的预算、query-trained CE、hop 分层和单跳控制结果见 `results/acl_extended_results.md`。

---

## 2 相关工作与研究边界

### 2.1 查询改写与假设文档

HyDE、Query2Doc 和 Rewrite-Retrieve-Read 通过生成假设文档、伪文档或改写查询，缓解疑问句与陈述句之间的表示差异。它们主要改变查询的表面表达或表示分布。本文关注更基础的情况：后续需求缺少前序答案所引入的实体变量。已有实验显示，三路查询改写在测量的 gap case 上仅消除 9.5%。

### 2.2 分解与迭代检索

Least-to-Most、Self-Ask 和 IRCoT 把复杂问题拆成子问题或推理步骤。IRCoT 的推理文本实际上能够隐式携带中间实体。本文与这些工作不是“是否分解”的竞争，而是将绑定从完整流程中隔离为一个可干预变量，并测量它对候选可达性的独立影响。

### 2.3 覆盖式选证与证据验证

SETR、PureCover 等工作在候选集内进行信息需求识别和集合选择；SURE-RAG 则把证据充分性用于验证或弃答。本文关注候选生成之前的问题：正确需求证据是否进入候选池。为此，本文将损失拆为候选可达性与选择效率。

### 2.4 图检索、NLI 与匹配器

知识图谱问答、GraphRAG、NLI 证据判定和 ColBERT 等分别提供结构推理、蕴含验证或多向量交互。它们可以作为后续实现组件，但不是本文的核心新颖性。本文不声称提出新的 OT、NLI 或多向量检索器，而是研究查询槽绑定作为第一阶段检索干预的因果作用。

---

## 3 问题定义

### 3.1 查询程序

将问题 $q$ 编译为查询程序 $P=(R,E)$：

- $R=\{r_1,\ldots,r_m\}$ 是带类型的原子需求；
- $E$ 是依赖边；
- $r_i$ 可以引用前序需求的答案，例如 `#1`。

绑定函数 $\beta$ 将前序答案填入依赖需求：

$$
\hat r_i = r_i[\beta].
$$

未绑定查询保留原问题或占位符；绑定查询显式包含后续证据所谈论的实体。

### 3.2 证据覆盖

设需求 $r_i$ 的金标证据为 $g_i$，证据集合为 $S$，则需求覆盖写为：

$$
\mathrm{cov}(S\mid\beta)=
\frac{1}{m}\sum_{i=1}^{m}
\max_{d\in S}\mathrm{ground}(\hat r_i,d).
$$

这里的 `ground` 表示证据 $d$ 是否满足绑定后的需求。

### 3.3 可达性与选择效率

设第一阶段候选集为 $C$，则候选可达性为：

$$
\mathrm{Reach}(C)=
\frac{1}{m}\sum_i
\mathbb{1}[g_i\in C].
$$

选择效率定义为最终覆盖相对于候选可达性的比例：

$$
\eta(S,C)=\frac{\mathrm{cov}(S)}{\mathrm{Reach}(C)}.
$$

因此有经验分解：

$$
\mathrm{cov}(S)=\mathrm{Reach}(C)\times\eta(S,C).
$$

可达性由 query slot、第一阶段 retriever 和语料决定；选择效率由 matcher 和集合选择规则决定。本文的核心干预作用于前者。

### 3.4 适用性判据

定义绑定缺口：

$$
B=\mathrm{cov}(\hat r\text{-top-}k)-\mathrm{cov}(q\text{-top-}k).
$$

实验表明，方法有效需要两个条件共同满足：

1. $B$ 足够大；
2. $k\ge m$，预算覆盖所有需求。

当 $k<m$ 时，逐需求取证可能提高需求覆盖率，却牺牲最后一跳证据，从而造成 EM 下降。

---

## 4 CovBind 方法

### 4.1 编译

规划器将问题编译为需求序列，并输出答案类型、依赖位置和关系描述。例如：

```text
r1: who plays old Ethan in A Dog's Purpose
r2: #1 spouse
```

若没有真实依赖边且问题不是 comparison 类型，则路由为单跳路径，避免无意义分解。

### 4.2 绑定与查询槽替换

执行第 $i$ 个需求时，将已抽取的前序答案填入需求：

```text
#1 spouse → Dennis Quaid spouse
```

检索和重排均使用绑定后的需求，而不是重新使用完整问题：

$$
C_i=\mathrm{Retrieve}(\hat r_i,\mathcal{C}),
\qquad
S_i=\mathrm{Rerank}(\hat r_i,C_i).
$$

该步骤不改变 encoder、reranker、语料或证据预算，改变的是证据与什么对象对齐。

### 4.3 逐需求选证

每个需求独立获得证据，最终合并并去重。若多个需求竞争证据预算，则使用覆盖增益进行贪心选择。已有消融显示，逐需求、全局 max 和覆盖贪心之间的差异不超过约 0.03；主要杠杆是查询槽，而不是聚合算子。

### 4.4 reader 与停止

reader 先定位证据句再抽取答案。多跳末跳与合成答案冲突时，使用类型化决策层选择末跳答案或全局合成答案。停止条件是所有需求被覆盖或达到预算上限，未覆盖需求集合可直接作为下一步检索目标。

### 4.5 不是新的 matcher

CovBind 使用 cross-encoder 作为工程组件。本文不把 CE 的结构、训练损失或 Role-OT 作为贡献。匹配器侧实验的作用是说明：在当前设置中，继续改变 matcher 的收益远小于改变 query slot。

---

## 5 实验设置

### 5.1 数据集

主数据集为 MuSiQue validation，使用前 200/400 题及 3,999 段 pooled corpus。MuSiQue 提供逐 hop 的 supporting paragraph 标签，适合做因果召回分析。外部验证包括 2WikiMultihopQA、HotpotQA 和 MultiHopRAG 新闻语料；SQuAD、TriviaQA 和 QuALITY 用作单跳控制。

### 5.2 模型与检索器

- BM25：$k_1=1.5,b=0.75$；
- dense：all-MiniLM-L6-v2；
- 对照：BGE-small、HyDE+MiniLM；
- cross-encoder：零样本 `ms-marco-MiniLM-L-6-v2` 或按数据集训练的需求 CE；
- reader/planner：`qwen3.6-27b`，temperature=0，关闭 thinking；
- chunk：40 tokens，stride=20。

所有训练 CE 按 qid 与评测题隔离。最终配置和结果索引见 `results/final_manifest.json`。

### 5.3 统计与协议

跨数据集实验并不共享完全相同的 hop 标注：MuSiQue 有原生逐 hop supporting paragraph，2Wiki 使用 evidence 三元组重建需求链，HotpotQA 和 MultiHopRAG 当前缓存主要提供段落级支持。因此，本文只把 MuSiQue 的 6.1–6.5 作为完整因果套件；2Wiki 的 query-slot 结果标为机制复验，HotpotQA/MultiHopRAG 的结果标为 passage-level reachability 和端点控制。协议审计见 `results/cross_dataset_protocol_audit.md`。

方法比较使用按题配对 bootstrap。因果主实验固定计划、reader、reranker、每跳 top-3 和候选池，只改变 `E13_QSLOT`。所有端点数字必须同时注明数据集、候选池、retriever、CE、query slot、计划来源、binding、reader 和停止策略。

---

## 6 实验结果

### 6.1 查询槽反事实

| 查询槽 | MuSiQue 200 EM | MuSiQue 400 EM |
|---|---:|---:|
| 整句问题 | 0.170 | 0.170 |
| 绑定后的需求 | **0.440** | **0.425** |
| 配对增益 | +0.270 [+0.210,+0.335] | **+0.255 [+0.210,+0.303]** |

400 题按跳分层：

| hop | 整句 | 绑定需求 |
|---|---:|---:|
| 2 | 0.197 | 0.520 |
| 3 | 0.189 | 0.377 |
| 4 | 0.079 | 0.276 |

该实验是全文最重要的证据：所有组件保持不变，只有 query slot 变化。

### 6.2 绑定恢复后续跳可达性

| hop | 未绑定 dense | 绑定 dense | 未绑定 BM25 | 绑定 BM25 |
|---|---:|---:|---:|---:|
| 1 | 0.915 | 0.970 | 0.795 | 0.945 |
| 2 | 0.615 | 0.935 | 0.425 | 0.940 |
| 3 | 0.673 | 0.921 | 0.554 | 0.891 |
| 4 | 0.538 | 0.923 | 0.256 | 0.897 |

第 1 hop 的变化来自需求聚焦；第 2–4 hop 的大幅恢复来自中间实体绑定。

### 6.3 可达性与选择效率

| retriever | candidate reachability | pointwise efficiency | coverage@3 |
|---|---:|---:|---:|
| BM25 | 0.610 | 0.472 | 0.288 |
| MiniLM | 0.760 | 0.558 | 0.424 |
| BGE-small | 0.765 | 0.563 | 0.431 |
| HyDE+MiniLM | 0.770 | 0.542 | 0.417 |

更强 encoder 抬高了候选可达性，但没有改变逐点选择效率。需求覆盖选择可以接近候选上限，说明候选生成和集合选择是两个不同问题。

### 6.4 覆盖是 reader 的瓶颈

在 gold 需求、gold binding、20 段池和 $k=2$ 的 2-hop 子集上：

| 选择器 | coverage | 全证据率 | EM |
|---|---:|---:|---:|
| whole question | 0.598 | 0.304 | 0.268 |
| bound requirement | **0.866** | **0.750** | **0.571** |

全证据时 EM 约 0.62–0.65，部分证据时约 0.155–0.169。证据是否齐全是主要变量。

### 6.5 训练分布影响小于查询槽

需求 CE 与问题 CE 使用相同训练题、正例和 hard negatives，仅改变训练 query：

| 训练 query | 推理 query | coverage@2 | 全证据率 | EM |
|---|---|---:|---:|---:|
| 完整问题 | 完整问题 | 0.608 | 0.400 | 0.23 |
| 完整问题 | 需求 | 0.684 | 0.440 | 0.35 |
| 需求 | 完整问题 | 0.509 | 0.310 | 0.25 |
| 需求 | 需求 | **0.738** | **0.590** | 0.35 |

推理时改变 query slot 带来大幅覆盖收益；训练 query 改变的覆盖收益为 +0.053，gold 计划下 EM 无显著变化。在真实 LLM 计划下，需求 CE 相对问题 CE 提升 +0.065。

### 6.6 多方法端到端比较

在已缓存且协议可追溯的方法结果上，得到如下对照。由于部分基线使用 zero-shot CE/BM25，而 CovBind 使用 dataset-domain CE/CoT reader，表中显式保留协议差异；严格因果结论仍以同 matcher 的 whole-query 反事实为准。

| 数据集 | Vanilla | Self-Ask | HyDE | IRCoT | CovBind | CovBind-whole-query |
|---|---:|---:|---:|---:|---:|---:|
| MuSiQue 200 | .175 | .075 | .205 | .330 | **.440** | .170 |
| MuSiQue 400 | — | — | — | .255 | **.425** | .170 |
| 2Wiki 200 | — | — | — | .535 | **.620** | — |
| HotpotQA 200 | .540 | — | — | .660 | **.680** | — |
| MultiHopRAG 200 | — | — | — | .465 | .530 | — |

MuSiQue 上 CovBind 超过 IRCoT；把 CovBind 的 query slot 换回整句问题后 EM 回落到 .170。2Wiki 与 HotpotQA 上接近 IRCoT，MultiHopRAG 只有已有 IRCoT/CovBind 端点结果。Self-Ask 与 HyDE 的完整缓存此前仅覆盖 MuSiQue 200；本轮补跑了 MuSiQue 400、2Wiki、HotpotQA、MultiHopRAG 的 Vanilla/Self-Ask。结果和数据格式限制见 `results/method_completion_report.md`。HyDE 仍只在 MuSiQue 200 纳入比较，因为跨数据集伪文档缓存尚未建立；缺失数据不填入伪造数字。完整协议说明见 `results/method_comparison_report.md`。新增 BM25/dense/hybrid 候选可达性方法对照见 `results/additional_methods_report.md`：dense 在低 k 上普遍提高 coverage，但 2Wiki/HotpotQA 在 k=10 饱和，MultiHopRAG 在 k=10 后仍存在不可达证据。

### 6.7 端到端与跨数据集

| 数据集 | Vanilla+CE | IRCoT+CE | CovBind | 相对 IRCoT |
|---|---:|---:|---:|---:|
| MuSiQue 200 | 0.195 | 0.315 | **0.440** | +0.125 |
| MuSiQue 400 | 0.190 | 0.302 | **0.425** | +0.123 |
| 2Wiki 200 | 0.475 | 0.615 | 0.620 | +0.005 |
| HotpotQA 200 | 0.625 | 0.665 | 0.680 | +0.015 |
| MultiHopRAG 200 | — | 0.465 | 0.530 | +0.065 |

本文不主张跨数据集端到端普适提升。稳定显著收益集中在 MuSiQue；2Wiki 和 HotpotQA 的 domain-CE 端点主要表现为持平。补充的 2Wiki 50 题 query-slot 配对实验中，绑定需求 coverage 从 .475 提升到 .800，hop recall 从 .480 提升到 .795，但该结果使用 top-20 候选和小规模 gold-binding 协议，应作为机制复验而非端到端主表。跨数据集 whole-query BM25 可达性曲线显示，2Wiki/HotpotQA 在 top-10 已饱和，而 MultiHopRAG 在 top-10 仍存在不可达证据，这与方法的结构性适用条件一致。

### 6.7 查询槽收益的跳数分层

对 MuSiQue 400 题的配对结果进一步按 hop 数分层：

| hop | n | 整句 EM | 绑定需求 EM | ΔEM | 95% CI |
|---|---:|---:|---:|---:|---|
| 2 | 198 | 0.197 | 0.520 | +0.323 | [+0.258,+0.394] |
| 3 | 122 | 0.189 | 0.377 | +0.189 | [+0.107,+0.270] |
| 4 | 76 | 0.079 | 0.276 | +0.197 | [+0.105,+0.289] |
| 全部 | 400 | 0.170 | 0.425 | +0.255 | [+0.210,+0.300] |

收益在 2-hop 最大，但 3-hop 和 4-hop 仍保持稳定正增益。单跳样本仅 4 题，不能据此做统计判断；单跳控制实验单独报告读者协议影响。

### 6.8 适用性判据

| 数据集/子集 | 绑定缺口 B | 预算 | 端到端变化 |
|---|---:|---:|---:|
| MuSiQue 2-hop | +0.268 | $k=m$ | +0.304 EM |
| 2Wiki | +0.040 | $k=m$ | +0.005 EM |
| MuSiQue 3-hop，$k=2$ | +0.173 | $k<m$ | −0.200 EM |
| MuSiQue 4-hop，$k=2$ | +0.184 | $k<m$ | −0.105 EM |

绑定缺口不是充分条件。预算不足时，覆盖选择可能丢掉末跳证据。

### 6.9 单跳控制与读者维度

现有单跳控制结果显示，SQuAD 每题 10 段池中，CoT reader 相对 Vanilla 为 0.715 对 0.665（+0.050）；TriviaQA 为 0.755 对 0.755。多跳 CovBind 的收益不能简单归因于 reader：MuSiQue 中接入 CoT reader 后仍主要由绑定查询槽带来增益，而 2Wiki 的 CoT reader 额外提升约 +0.025。全局 2000 段池上的单跳路由实验会受到错误多跳规划影响，因此不作为主性能表。

### 6.10 负结果

1. 查询改写消除 9.5% 的 gap case；
2. LLM 绝对充分性判断在 582 个真实 gap case 上低于 base CE；
3. 答案似然首跳有效、第二跳失效，gold 段平均边际为 −0.113，干扰段为 +0.096；
4. 绑定门控、共识绑定、回滚、前瞻绑定和多种 join 未形成稳定跨数据集收益；
5. Role-OT/结构事实 grounding 作为工程探索未通过初筛，因此不纳入主方法。

---

## 7 误差分析与讨论

### 7.1 绑定错误会沿链传播

金标绑定时后续 hop Recall@50 可达 0.89–0.94；预测绑定后端到端收益明显下降，第 4 hop recall 可回落至 0.359。自动绑定约消耗 65% 的 oracle 绑定收益。绑定质控实验显示，逐字存在、CE margin、类型兼容和前瞻检索性都不能稳定识别语义漂移。

### 7.2 为什么 matcher 不是核心

当候选根本不包含目标证据时，任何 reranker 都没有作用。E2a 与 E10 显示，更强编码器和蒸馏 matcher 的收益有限；而固定 matcher、仅改变 query slot 即带来 +0.255 EM。这个结论不是“matcher 无用”，而是“matcher 的作用受 query slot 设定的候选上限约束”。

### 7.3 结果为何在数据集间不同

MuSiQue 的组合结构、较大的 pooled corpus 和自然的 hop query 使 $B$ 较大。2Wiki/HotpotQA 的候选池较小，整句问题已经足以召回支持段，因此绑定的边际空间较小。该解释是可测的适用性判据，而不是按数据集名称贴标签。

### 7.4 研究边界

本文没有解决：

- 计划错误的可靠在线检测；
- 错误绑定的语义验证；
- 完整开放域 Wikipedia 上的绝对性能；
- 跨语言和专业领域迁移；
- reachability × selection efficiency 的参数化理论界。

---

## 8 结论

本文通过单变量查询槽干预，证明多跳 RAG 中一个常被归入“语义匹配”的失败实际上来自查询变量未绑定。将前序答案绑定到后续需求后，正确证据的候选可达性显著恢复；覆盖选择则负责在候选内部利用这些证据。该方法的收益依赖两个可测条件：绑定缺口足够大、证据预算覆盖所有需求。

本文的核心结论不是更强的 matcher 可以解决所有 query–evidence gap，而是：

> **在多跳检索中，先让查询指向正确的对象，再讨论如何匹配。**

---

## 参考文献

本文完整参考文献列表可由现有调研文件 `survey.md`、`top10.md` 和 `papers.csv` 生成。正式投稿前需按 ACL Anthology 格式补齐 BibTeX，至少包括：DPR、HyDE、Query2Doc、Rewrite-Retrieve-Read、Self-Ask、Least-to-Most、IRCoT、Adaptive-RAG、SETR、PureCover、SURE-RAG、CRAG、Self-RAG、RARR、GraphRAG、ColBERT 和 MuSiQue。

---

## 附录 A：复现清单

最终协议索引：

```text
results/final_manifest.json
```

核心查询槽反事实：

```bash
CE_DATASET_URL=http://10.52.101.89:8811 \
CE_DATASET=musique E13_COT=1 \
python3 -m src.e13_exec --n 200 --workers 8 \
  --modes select --data musique200.jsonl

CE_DATASET_URL=http://10.52.101.89:8811 \
CE_DATASET=musique E13_COT=1 \
E13_QSLOT=question E13_SUFFIX=_qslotq \
python3 -m src.e13_exec --n 200 --workers 8 \
  --modes select --data musique200.jsonl
```

覆盖诊断：

```bash
python3 -m src.e23_eval_reqce --n 100 \
  --model dist/reqce/model_hop \
  --modes whole,seq,global,cover \
  --plan gold --read
```

---

## 附录 B：结果协议字段

每个结果行应记录：

```text
commit
数据集与切分
候选池
retriever
CE checkpoint
query slot
plan source
binding source
k / budget
reader prompt
stop policy
random seed
result file
```

旧版单配置报告和 E24/E26 探索结果不应与 final endpoint table 混合。
