# 表格补全与增强结论的实验建议

## 已补全的统一端到端表

| dataset | Vanilla | Self-Ask | Query Rewrite | HyDE | Decomposition | IRCoT | CovBind |
|---|---:|---:|---:|---:|---:|---:|---:|
| MuSiQue-200 | .155 | .055 | .155 | .220 | .165 | .285 | .440* |
| MuSiQue-400 | .128 | .060 | 待运行 | 待运行 | 待运行 | .248 | .425* |
| 2Wiki-200 | .375 | .265 | .395 | .390 | .425 | .510 | .620* |
| HotpotQA-200 | .530 | .295 | .525 | .530 | .545 | .650 | .680* |
| MultiHopRAG-200 | .480 | .565 | .515 | .540 | .555 | .485 | .530* |

`*` CovBind 使用 domain-CE/原有 reader 协议；其余新行使用 H3/Qwen3.8-27B + BM25 + 统一 reader，不能作严格同模型单变量比较。

MuSiQue-400 的 Query Rewrite/HyDE/Decomposition 仍标为待运行，避免把 200 题结果替换或复制到 400 题。

## 可执行的稳健性实验

1. **同模型全矩阵**：在 H3/Qwen3.8 上补跑 MuSiQue-400 的三个额外方法，并重新跑 CovBind/whole-query；这是最优先缺口。
2. **端点交叉验证**：固定 qid，将 8000–8003 与 8004–8007 分组运行，报告端点组间差异。
3. **随机种子重复**：对每个方法重复 3 次，固定检索结果，只改变生成温度/端点分配。
4. **预算曲线**：top-k={1,3,5,8,10}，比较 evidence coverage 与 EM，验证收益不是预算差异。
5. **oracle plan vs predicted plan**：同一 reader 下分别使用 gold plan、LLM plan、无 plan，分离规划误差和 binding 贡献。
6. **反事实 binding**：whole query、unbound hop、gold binding、predicted binding 四臂配对比较。
7. **跨域外推**：按2-hop/3-hop/4-hop及问题类型分层，报告效果是否集中在长链问题。
8. **reader ceiling**：完整 evidence、缺一跳 evidence、随机缺证据三种输入，量化 coverage-reader bottleneck。
9. **错误类型分析**：把失败分成 candidate unreachable、wrong selection、reader extraction、final join 四类，并报告占比。
10. **统计稳健性**：paired bootstrap、McNemar、Holm correction，避免多方法多数据集比较造成偶然显著。
