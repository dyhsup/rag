# MuSiQue-400 额外端到端方法补全

| 方法 | n | 唯一 qid | 空预测 | 服务错误 | EM | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Query Rewrite | 400 | 400 | 1 | 0 | .160 | .261 |
| HyDE | 400 | 400 | 0 | 0 | .223 | .319 |
| Decomposition | 400 | 400 | 2 | 0 | .168 | .264 |

空预测按失败计入总体指标；没有删除这些题，也没有用其他方法结果替代。逐题文件：

- `results/h3e2e_musique400_rewrite.jsonl`
- `results/h3e2e_musique400_hyde.jsonl`
- `results/h3e2e_musique400_decomp.jsonl`

## 同模型 CovBind 状态

当前 CovBind 主结果使用既有 domain-CE/reader 管线，和 H3/Qwen3.8 额外 baseline 并非同一模型协议。为了避免伪造“同模型 CovBind”，本轮不把旧 CovBind 数字改标成 H3 结果。严格替代方案已经记录：需要为 H3 服务重跑 plan、binding、CE/selection 和 whole-query counterfactual，并生成逐题 trace 后才能加入同模型表。
