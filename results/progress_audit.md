# 项目进度审计与 Query–Evidence 实验路线

## 1. 当前判断

项目已经完成：文献调研、受控改写探针、自然语料覆盖分解、检索器天花板对比、查询槽反事实、自动绑定误差传播、端到端跨数据集复验、蒸馏/匹配侧负结果扫描。

当前最稳的因果结论不是“CE 不够强”，而是：**整句问题被错误地作为单段证据的匹配查询；把查询槽替换为绑定后的需求后，后续跳的证据可达性显著恢复。**

## 2. 已完成事实

| 层次 | 已确认结果 | 解释 |
|---|---|---|
| 第一段/后续跳召回 | 未绑定 dense 约 0.915→0.538；绑定后约 0.970/0.935/0.921/0.923 | 后跳缺少中间实体是主要崩塌源 |
| 自然语料覆盖 | BM25 可达性 .610；MiniLM/BGE/HyDE .760–.770 | 更换编码器不能消除第一段天花板 |
| 覆盖选择 | CE 覆盖 @3 .677，对 dense 逐点 .424；约达 oracle 90% | 逐点 rerank 不是核心杠杆 |
| 读者上限 | 全证据 EM .62–.65；部分证据 .155–.169 | evidence coverage 是端到端瓶颈 |
| 查询槽反事实 | 400 题 paired ΔEM=.255，bootstrap 95% CI [.210,.300] | 同计划、同读者、同 CE，只换 whole→bound query slot |
| 按跳反事实 | 2-hop +.323；3-hop +.189；4-hop +.197；单跳 n=4 无差异 | 深链叙事有支持，但单跳必须路由 |
| E10真实口径 | base CE closure@3=.198；蒸馏=.227；LLM=.170 | 匹配侧蒸馏有效但幅度小 |
| 跨集 | MuSiQue 稳定增益；2Wiki/Hotpot 多数持平；MHR 边缘正 | 方法受数据结构与绑定缺口制约 |

## 3. 协议风险

仓库并存多个合法但不可直接混合的协议：域内 CE、零样本 CE；BM25、混合检索；CoT reader、普通 reader；末跳、合成、决策层停止；200/400 题及不同候选池。`.440/.425/.430/.428/.435` 并非必然矛盾，但当前没有统一 manifest。

`results/e4_report*.md` 是旧版/单配置报告，不应和 E13/G1 的最终配置混入同一主表。正式论文主表必须逐行记录 commit、数据切分、候选池、retriever、CE checkpoint、query slot、plan 来源、binding、k、reader、停止策略与结果文件。

## 4. 十个真正跳出“绑定 + 普通 CE”的方向

1. **Role-OT**：实体/关系/属性槽与证据事实做带关系方向和角色守恒的最优传输。
2. **Late Interaction**：ColBERT 多向量 MaxSim，保留需求 token–证据 token 对齐矩阵，用跨 hop hard negative 训练。
3. **Directional NLI**：证据蕴含需求、需求蕴含证据、矛盾三路判别，区分主题相关与事实支持。
4. **MIL-Span**：把段落视为句包，学习多句共同支持需求，避免把单句删除监督重新包装。
5. **Entity-Path**：实体/关系图中计算绑定实体到目标关系的有向路径与类型一致性。
6. **Complementarity Pair Encoder**：学习跨需求证据的互补向量，不做已有 E29 式联合集合分类。
7. **Counterfactual Invariance**：无关实体/句子变换保持分数，关系否定或关键实体替换使分数下降。
8. **Retriever-Agreement Fact Clusters**：把不同检索器结果映射到事实簇，学习共识与互补，不是简单并集。
9. **Domain-Invariant Relation Prototypes**：关系、实体类型、操作原型匹配，MuSiQue 训练后跨域测试。
10. **Sufficiency×Necessity Grid**：独立建模“段落足以支持需求”和“移除后支持崩溃”，用四象限校准主题相关与真正支撑。

## 5. 统一最小筛选协议

- gold 需求、固定绑定结果、固定候选池、固定 k=2/3、固定 reader；先 n=100。
- 同题配对 bootstrap 95% CI；同时报 candidate reachability、selection efficiency、coverage、full-evidence、EM/F1。
- 通过线：相对需求 CE coverage 至少 +3pp 且 CI 不跨 0；端到端 EM 至少 +2pp。
- 任何方向若 MuSiQue 有效但 2Wiki/Hotpot 任一 EM 回退超过 1pp，标为数据集特化；若首轮不达线立即停止。
- 优先顺序：Role-OT → Directional-NLI → Entity-Path → Late Interaction → MIL-Span。

## 6. 下一步决策

第一步不是继续堆叠融合，而是冻结主表和协议 manifest；第二步只押一次公平的结构化匹配复验。首个 Role-OT-style 原型已完成：MuSiQue gold-plan 100 题、20 段池、k=2 上，需求 CE coverage=.7308，结构 grounding=.5025，CE+结构=.7358；未达到 +3pp 通过线，因此该低成本原型停止扩展。结果见 `results/e24_role_ot_report.md`。这不否定“需求变量—角色—关系方向的第一阶段 grounding”作为研究问题，但说明手工关系词表/表面实体启发式不足以替代训练 CE。下一轮若继续，只做一次独立训练的结构感知 matcher 或事实抽取器公平复验；E26 已用 `k=m` 做了 20 题结构事实 grounding smoke test，hop coverage=.2292、full-hop=.0000，对比需求 CE .8083/.6500，说明离线模板工程实现不可用。若不再投入独立训练审计，研究应收束为“查询槽因果定位 + 适用性判据 + 系统负结果”，不再把匹配器作为第二主线。
