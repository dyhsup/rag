# Query–Evidence 下一轮实验方案

> 目标：在不重复“绑定查询槽 + 普通 CE”结论的前提下，寻找真正的 query–evidence 匹配方法创新。
> 统一协议：MuSiQue gold plan，候选池固定，k=2/3，需求 CE 为主基线；每项先 n=100，配对 bootstrap 95% CI。

## 当前进度冻结摘要

- 主因果结果：MuSiQue 400 题，绑定需求查询槽相对整句查询 EM +0.255；本地复核 paired n=400，bootstrap 95% CI [0.210, 0.300]。
- 按跳增益：2-hop +0.323、3-hop +0.189、4-hop +0.197；单跳 n=4 无差异。
- 覆盖瓶颈：全证据时 reader EM 约 0.62–0.65，部分覆盖约 0.155–0.169。
- E10 真实口径：base CE closure@3=.198，蒸馏=.227，LLM=.170。
- 跨集：MuSiQue 稳定；2Wiki/HotpotQA 多数持平；方法需按结构条件路由。

## 十个新方向

| 编号 | 方案 | 核心对象 | 最小实验 | 通过线 | 主要风险 |
|---|---|---|---|---|---|
| A | Role-OT | 实体/关系/答案类型的最优传输与角色守恒 | 抽取需求和段落事实，gold 需求、20 段池、k=2 | coverage +3pp 且显著，EM +2pp | 事实解析错误 |
| B | Late-Interaction | query token 与 evidence token 多向量 MaxSim，不压成单一 CE 分数 | MiniLM/ColBERT 小模型，跨 hop hard negative | coverage +5pp，EM +3pp | 仍受词面偏置影响 |
| C | Directional-NLI | evidence→requirement 蕴含、反向蕴含、矛盾三路分数 | 300 对人工标注支持/主题/矛盾/无关 | 支持 PR-AUC>.80，coverage +3pp | 长句隐含关系误判 |
| D | MIL-Span | 段落内多句协同支持，而非单句删除 | 句包编码 + set pooling，段级弱监督 | 长段子集 coverage +5pp | 与 E31 删除监督混淆 |
| E | Entity-Path | 实体图中从绑定实体到目标关系的有向路径 | oracle 实体抽取→预测实体抽取 | oracle +8pp，预测 +3pp | 实体链接级联错误 |
| F | Counterfactual Invariance | 对无关实体/无关句替换保持分数，对关系破坏敏感 | 三类反事实变换，固定 CE 候选 | 无关稳定性>.90，关系破坏 AUC>.80 | 只学到编辑痕迹 |
| G | Relation Prototype | 关系、类型、操作原型的域不变匹配 | MuSiQue 训练，2Wiki/Hotpot 零样本 | MuSiQue 不降，跨集各 +3pp | 原型过粗 |
| H | Evidence Graph Matching | 需求图与证据事实图做节点/边匹配，不做集合标量分类 | 2-hop/3-hop gold graph，beam path matching | bridge coverage +4pp | 图抽取成本高 |
| I | Query–Evidence Co-training | query 需求表示与 evidence 事实表示交替对齐，使用错绑定 hard negative | 训练集 [400:1000]，独立 dev/eval | closure@3 +3pp，跨协议稳定 | 训练泄漏/过拟合 |
| J | Span-Bridge Alignment | 先找关系触发词和实体跨度，再对齐关系方向 | 句级 span 标注/弱标注，固定 reader | relation 子集 R@3 +5pp | 对隐式关系无效 |

## 推荐执行顺序

1. A Role-OT；2. C Directional-NLI；3. E Entity-Path；4. B Late-Interaction；5. D MIL-Span；6. F 反事实一致性；7. G 原型匹配；8. H 证据图；9. I 联合训练；10. J span bridge。

所有方案必须同时报告：candidate reachability、selection efficiency、coverage@k、full-evidence rate、EM/F1，并按 2/3/4-hop 和 MuSiQue/2Wiki/HotpotQA 分层。若相对逐需求 CE coverage 未达到 +3pp 且 CI 跨 0，立即停止该方向；若 MuSiQue 有效但 2Wiki/HotpotQA 回退超过 1pp，标记为数据集特化而非通用方法。

## 不再重复

已失败或已充分覆盖：max-min/soft-min 集合聚合、答案空间信号、集合判别、删除监督、CMU 答案似然、绑定门控/共识/回滚/前瞻、简单 query rewrite、BM25+dense 简单并集。

## 主表冻结要求

每个结果行必须记录：commit、数据切分、候选池、retriever、CE checkpoint、query slot、plan 来源、binding 状态、k、reader prompt、停止策略、随机种子和结果文件。旧 `e4_report*.md` 标为 legacy，不与 E13/G1/E23 结果直接混表。
