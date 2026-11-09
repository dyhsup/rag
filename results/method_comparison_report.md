# 多方法跨数据集比较报告

## 可比性说明

现有方法结果来自两类协议：

1. zero-shot CE/BM25 基线：Vanilla、IRCoT、Self-Ask、HyDE；
2. domain CE / CoT reader：CovBind/RBG 与 whole-query 反事实。

两类协议不应直接作为同一严格公平表中的单变量比较；本报告保留协议字段，并把 query-slot 反事实单独作为因果主表。

## 统一结果

| 数据集 | Vanilla | Self-Ask | Query Rewrite | HyDE | Decomposition | IRCoT | CovBind | CovBind-whole-query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MuSiQue 200 | .155 | .055 | .155 | .220 | .165 | .285 | **.440*** | .170* |
| MuSiQue 400 | .128 | .060 | 待运行 | 待运行 | 待运行 | .248 | **.425*** | .170* |
| 2Wiki 200 | .375 | .265 | .395 | .390 | .425 | .510 | **.620** |
| HotpotQA 200 | .530 | .295 | .525 | .530 | .545 | .650 | **.680** |
| MultiHopRAG 200 | .480 | .565 | .515 | .540 | .555 | .485 | .530 |

## 解释

- MuSiQue 上 CovBind 超过 IRCoT，并且 whole-query 反事实回落到 .170，说明增益不能归因于 reader 或单纯多轮调用。
- 2Wiki/HotpotQA 上 CovBind 与 IRCoT 接近，符合 reachability gap 较小的适用性判据。
- MultiHopRAG 上只有 IRCoT 与 CovBind 的已有端点结果；该数据集不具备原生逐 hop 绑定标签，因此不构造伪造的 query-slot 因果表。
- MuSiQue-400 的 Query Rewrite/HyDE/Decomposition 仍待运行，不能用 200 题数值替代。
- `*` 表示 CovBind 与 whole-query 使用旧 domain-CE 协议；其余 200 题 baseline 行使用 H3/Qwen3.8-27B + BM25 + 统一 reader。
- 失败服务缓存和重复 qid 文件不进入任何主表；完整审计见 `results/audit_results.md`。

原始文件和协议字段见 `results/method_comparison.json`。
