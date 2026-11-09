# 缺失方法结果补全报告

## 已补全

| 数据集 | 新增方法 | n | EM | F1 | 原始结果 |
|---|---|---:|---:|---:|---|
| MuSiQue-400 | Vanilla | 400 | .020 | .038 | `e4_musique400_extended_vanilla.jsonl` |
| MuSiQue-400 | Self-Ask | 400 | .038 | .083 | `e4_musique400_extended_selfask.jsonl` |
| 2Wiki-200 | Vanilla | 200 | .015 | .018 | `e4_wiki2_extended_vanilla.jsonl` |
| 2Wiki-200 | Self-Ask | 200 | .015 | .018 | `e4_wiki2_extended_selfask.jsonl` |
| HotpotQA-200 | Self-Ask | 200 | .540 | .643 | `e4_hotpot_extended_selfask.jsonl` |
| MultiHopRAG-200 | Vanilla | 200 | .000 | .000 | `e4_mhr_extended_vanilla.jsonl` |
| MultiHopRAG-200 | Self-Ask | 200 | .000 | .000 | `e4_mhr_extended_selfask.jsonl` |

## 仍缺失且不填伪造数字

- 2Wiki/Hotpot/MultiHopRAG 的 HyDE：现有 `g4_hyde.py` 固定读取 MuSiQue 的 `hyde_pseudo.jsonl`，没有对应数据集伪文档缓存；跨数据集直接复用会造成 query 泄漏/协议错误。
- 2Wiki/Hotpot/MultiHopRAG 的 CovBind-whole-query EM：现有端到端缓存只有 CovBind domain-CE 正向端点；whole-query 反事实已有 MuSiQue 200/400。跨集已有 passage-level whole-vs-bound coverage 或 reachability，但没有同 reader/同 CE 的完整 EM 文件，故不把 coverage 代写成 EM。
- MultiHopRAG 的 CovBind-whole-query EM：该缓存没有原生逐 hop dependency plan，不能把平面 evidence list 强行转成绑定流水线。
- MuSiQue-400 HyDE：现有 HyDE pseudo cache 只覆盖 MuSiQue 200 qid；扩展生成需重新调用 LLM，当前不纳入未完成结果。

## 解释

部分新增 zero-shot Vanilla/Self-Ask 数值异常低（例如 2Wiki .015、MHR 0），这反映现有 E4 reader/数据格式在这些数据集上不匹配，不能作为“方法普遍很差”的结论。论文中应把这些结果标为 exploratory，并以 IRCoT 与 CovBind domain-CE 的可比端点、以及单变量 query-slot 反事实为主证据。
