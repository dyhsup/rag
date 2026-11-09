# 额外端到端方法实验报告

## 实验设计

实现了三个额外端到端方法：

- Query Rewrite：LLM 将问题改写为检索 query；
- HyDE：LLM 生成 hypothetical passage，再检索；
- Decomposition Retrieval：LLM 将问题拆成多个可检索 query，合并证据后阅读。

入口：`src/extended_e2e_methods.py`。

## 运行范围

每个数据集 50 题：

- MuSiQue：`e2e_musique50_*`
- 2Wiki：`e2e_wiki250_*`
- HotpotQA：`e2e_hotpot50_*`
- MultiHopRAG：`e2e_mhr50_*`

## 运行状态

本轮运行期间两个 LLM endpoint 均返回连接失败。逐题记录的错误为：

```text
HTTPConnectionPool(host=10.94.199.216/217, port=8000): Failed to establish a new connection
```

因此四个数据集上的 rewrite/HyDE/decomp EM=0.0 是**服务不可用导致的空预测**，不是方法性能结果，禁止纳入方法比较表或论文主表。

原始诊断文件仍保留：

- `results/e2e_musique50_{rewrite,hyde,decomp}.jsonl`
- `results/e2e_wiki250_{rewrite,hyde,decomp}.jsonl`
- `results/e2e_hotpot50_{rewrite,hyde,decomp}.jsonl`
- `results/e2e_mhr50_{rewrite,hyde,decomp}.jsonl`

## 结论

额外方法的代码路径已建立，但有效 EM/F1 尚未获得。待 LLM 服务恢复后，应清理这些失败缓存或使用新 tag 重跑，并要求记录 `error_count=0` 后才允许进入比较表。
