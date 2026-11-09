# ACL 主会投稿前检查清单

## 当前已完成

- [x] 核心问题收束为 query-slot causal diagnosis
- [x] 主因果实验及 200/400 题结果
- [x] 候选可达性与选择效率分解
- [x] MuSiQue、2Wiki、HotpotQA、MultiHopRAG 跨数据集结果
- [x] 绑定缺口与预算充分性判据
- [x] 查询改写、答案似然、LLM 判断、绑定质控和 matcher 扫描负结果
- [x] Legacy/exploratory 结果标记
- [x] 最终协议 manifest
- [x] 中文 ACL 风格论文初稿

## 投稿前仍需人工完成

1. 将 `survey.md` / `top10.md` / `papers.csv` 的论文元数据整理成 ACL BibTeX，并在正文插入标准引用。
2. 从最终 JSONL 生成正式论文图：
   - hop Recall@50：绑定前后；
   - coverage vs budget；
   - reachability × selection efficiency；
   - MuSiQue/2Wiki/HotpotQA/MultiHopRAG 端点对照。
3. 审核所有数字的协议字段，尤其区分：
   - query-slot causal endpoint；
   - domain-CE endpoint；
   - zero-shot CE endpoint；
   - hybrid retrieval endpoint。
4. 明确标注 gold-plan/gold-binding 诊断实验与 predicted-binding 端到端实验。
5. 补充正式显著性表：每个主比较给出 paired bootstrap CI 和样本数。
6. 检查“IRCoT+CE”是否与各表使用完全相同的 reader、候选池和停止策略。
7. 复现环境中记录 LLM 服务版本、checkpoint hash、数据 hash 和 commit。
8. 运行完整命令并确认外部服务仍可访问后再投稿。

## 审稿风险

- 主贡献可能被认为是 IRCoT/Self-Ask 已隐含完成的绑定；必须突出单变量因果隔离与 reachability 分解。
- 绝对 EM 只在 MuSiQue 显著，不能宣称通用性能提升。
- gold binding 诊断存在上界性质，不能与真实流水线结果混写。
- 自动计划与绑定错误仍未解决，需放入主要局限而非附录。
- “次模覆盖保证”只对给定 grounding 假设成立，不能写成端到端性能保证。
- 当前中文稿的参考文献尚未转成 ACL 标准引用格式。
