# 多方法跨数据集比较报告

## 可比性说明

现有方法结果来自两类协议：

1. zero-shot CE/BM25 基线：Vanilla、IRCoT、Self-Ask、HyDE；
2. domain CE / CoT reader：CovBind/RBG 与 whole-query 反事实。

两类协议不应直接作为同一严格公平表中的单变量比较；本报告保留协议字段，并把 query-slot 反事实单独作为因果主表。

## 统一结果

| 数据集 | Vanilla | Self-Ask | HyDE | IRCoT | CovBind | CovBind-whole-query |
|---|---:|---:|---:|---:|---:|---:|
| MuSiQue 200 | .175 | .075 | .205 | .330 | **.440** | .170 |
| MuSiQue 400 | — | — | — | .255 | **.425** | .170 |
| 2Wiki 200 | — | — | — | .535 | **.620** | — |
| HotpotQA 200 | .540 | — | — | .660 | **.680** | — |
| MultiHopRAG 200 | — | — | — | .465 | .530 | — |

## 解释

- MuSiQue 上 CovBind 超过 IRCoT，并且 whole-query 反事实回落到 .170，说明增益不能归因于 reader 或单纯多轮调用。
- 2Wiki/HotpotQA 上 CovBind 与 IRCoT 接近，符合 reachability gap 较小的适用性判据。
- MultiHopRAG 上只有 IRCoT 与 CovBind 的已有端点结果；该数据集不具备原生逐 hop 绑定标签，因此不构造伪造的 query-slot 因果表。
- Self-Ask 与 HyDE 仅在 MuSiQue 200 的 zero-shot 配置中有完整缓存结果，不能扩写为所有数据集的缺失结果。

原始文件和协议字段见 `results/method_comparison.json`。
