# 跨数据集 Query–Evidence 扩展实验

## 1. 查询槽反事实：2Wiki 小规模严格配对

协议：2Wiki dev 缓存，50题，20段池，gold plan/gold binding，需求 CE=`models/ce_5dataset/models/wiki2`，k=2，top-20候选。该实验与 MuSiQue 的 6.1 对应，但数据集支持结构和候选池不同。

| 指标 | 整句问题 | 绑定需求 | 差值 |
|---|---:|---:|---:|
| 段落 coverage | 0.475 | 0.800 | +0.325 |
| 全证据率 | 0.100 | 0.600 | +0.500 |
| hop recall | 0.480 | 0.795 | +0.315 |

配对 coverage delta 的 95% bootstrap CI 为 `[+0.255,+0.400]`。

结果文件：`results/cross_slot_wiki2_50.json`。

**协议边界：**这是 50 题小规模、top-20 候选的验证；它与端到端 2Wiki 200 题的 domain-CE 结果不是同一协议。它说明 query-slot effect 在一部分 2Wiki 样本上可观察，但不能据此宣称跨数据集端到端普适。

## 2. Whole-query BM25 candidate reachability curves

该实验不声称绑定效果；它测量各数据集在整句问题 BM25 下，gold supporting passages 随候选 k 的可达性。每题200题，k∈{1,3,5,10,20,50}。

| 数据集 | any@1 | coverage@1 | any@3 | coverage@3 | any@5 | coverage@5 | any@10 | coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | .800 | .340 | .970 | .623 | .985 | .744 | 1.000 | 1.000 |
| HotpotQA | .710 | .355 | .905 | .663 | .970 | .773 | 1.000 | 1.000 |
| MultiHopRAG | .805 | .325 | .840 | .666 | .850 | .766 | .860 | .844 |

结果文件：

- `results/cross_reach_wiki2.json`
- `results/cross_reach_hotpot.json`
- `results/cross_reach_mhr.json`

**解释：**2Wiki 和 HotpotQA 的小候选池在 top-10 已达到 supporting-passage coverage 饱和，说明 whole-query 的候选可达性缺口很小；MultiHopRAG 新闻池在 top-10 后仍有不可达样本，且证据有效性/题型结构与逐跳 MuSiQue 不同。该结果支持“适用性由可测 reachability gap 决定”，而不是简单按数据集名称判断。

## 3. 需求 CE 与问题 CE

Gold plan、100题、k=2 的需求覆盖：

| matcher | whole | seq | global | cover |
|---|---:|---:|---:|---:|
| 问题 CE `model_q` | .531 | .713 | .681 | .712 |
| 需求 CE `model_hop` | — | **.731** | — | — |

该结果支持：训练 query 分布有有限作用，但推理时将 query slot 换成需求仍是主要变化。
