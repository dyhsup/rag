# 反伪造审计结论

## 结论

本审计没有发现“手工编造逐题预测”的证据；主表数字均可追溯到项目中的逐题 JSONL 或明确的汇总文件。与此同时，审计发现若干结果不能作为有效性能证据，已从主表语义中降级或明确标注。

## 已排除

1. `e2e_*` 与早期 `h3_e2e_*` 中包含 endpoint connection failure 的文件：这些是服务失败诊断，不是性能结果。
2. `e8_mined.jsonl`、`e8_scored.jsonl`、`e5_claim.jsonl`、`judge.jsonl`、`e12_forced.jsonl` 等存在大量重复 qid 的内部/中间文件：不用于端到端主表。
3. 首次 decomposition 运行的 `KeyError("queries")` 结果：只保留修复后的 `*_fix` 结果。
4. 空预测比例很高的旧 E4 Vanilla/Self-Ask 文件：保留作历史诊断，不用作 H3/Qwen3.8 主比较。

## 主表保留原则

- 每道题必须有唯一 qid；
- 必须有非空预测，或空预测必须显式计入失败且无服务错误；
- 必须能定位到逐题 JSONL；
- 必须记录模型、端点、reader、retriever、候选预算和计划来源；
- 不同 domain-CE 与 H3/Qwen3.8 协议不进行未经标记的直接显著性比较。

## 当前主表状态

| 数据块 | 状态 |
|---|---|
| H3/Qwen3.8 200题 Vanilla/IRCoT/Self-Ask | 保留 |
| H3/Qwen3.8 200题 Query Rewrite/HyDE/Decomposition | 保留；逐题文件无服务错误，少量空预测按失败计入 |
| H3/Qwen3.8 MuSiQue-400 额外三方法 | 待运行，不填充 |
| 旧 domain-CE CovBind | 保留，但标记为不同协议 |
| endpoint failure 输出 | 排除 |
| 重复 qid 中间文件 | 排除 |

逐文件 hash、qid、空预测和错误统计见 `results/audit_results.md` 与 `results/audit_results.json`。
