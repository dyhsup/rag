# 额外方法与检索器对比

## 端到端基线

已补全的 Vanilla/Self-Ask/IRCoT/CovBind 表见 `results/method_comparison.md`。本轮不把缺少独立 pseudo-document cache 的跨集 HyDE 结果填成数字。

## BM25 / dense / hybrid 候选可达性

这是 passage-level reachability，不是端到端 EM；候选池和 gold supporting passage 定义与 `cross_retriever_reach.py` 一致。

| 数据集 | retriever | cov@1 | cov@3 | cov@5 | cov@10 |
|---|---|---:|---:|---:|---:|
| 2Wiki | BM25 | .340 | .623 | .744 | 1.000 |
| 2Wiki | dense | .396 | .665 | .781 | 1.000 |
| 2Wiki | hybrid | .395 | .675 | .783 | 1.000 |
| HotpotQA | BM25 | .355 | .663 | .773 | 1.000 |
| HotpotQA | dense | .395 | .713 | .803 | 1.000 |
| HotpotQA | hybrid | .408 | .783 | .853 | 1.000 |
| MultiHopRAG | BM25 | .325 | .666 | .766 | .844 |
| MultiHopRAG | dense | .339 | .751 | .830 | .860 |
| MultiHopRAG | hybrid | .337 | .750 | .817 | .855 |

## 结论

- dense 在三套数据上提升低 k 候选覆盖；HotpotQA hybrid 在 k=3/5 的提升最明显。
- 2Wiki/HotpotQA 在 k=10 饱和，说明其端到端 query-slot 余量有限。
- MultiHopRAG 在 k=10 后仍低于 0.86，说明该新闻语料仍有不可达证据；但 hybrid 未超过 dense，简单并集不是稳定方案。
- 这些是候选可达性结果，不能替代 reader EM，也不等于 CovBind 方法对比。

原始文件：`results/retriever_wiki2.json`、`results/retriever_hotpot.json`、`results/retriever_mhr.json`。
