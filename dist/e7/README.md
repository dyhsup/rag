# E7 训练包：把 LLM 的 gap 判定蒸馏成 cross-encoder

自包含，只依赖 torch + transformers，不需要本项目的其它代码。

## 文件

- `train_gpu.py`：训练 + 评测
- `config.json`：超参
- `pairs.jsonl`：5400 条 `(query, chunk, LLM 分数 0-10, is_gold, dense_rank)`
- `requirements.txt`

## 运行

```bash
pip install -r requirements.txt
python train_gpu.py                 # 读 config.json
python train_gpu.py --smoke         # 64 条、1 epoch，先验证环境
```

默认：`ms-marco-MiniLM-L-6-v2` 初始化，3 epochs，lr 2e-5，batch 64，max_len 128，
按**问题 id 分组切分**（70/30），避免同一问题的样本跨集泄漏。

## 预期

- T4：约 10–20 s/epoch，总计 1–3 分钟
- A10/3090：约 3–5 s/epoch
- CPU（仅 smoke 用）：会慢，不要跑全量

## 输出

- `./model/`：训练后的 cross-encoder
- `metrics.json`：

```json
{
  "base_msmarco":  {"reachable_cases": 64, "closure@3": 0.56, "gap_cases": 38, "gap_closure@3": 0.56},
  "distilled":     {"reachable_cases": 64, "closure@3": 0.73, "gap_cases": 38, "gap_closure@3": 0.73}
}
```

基线数字（本地 CE 与 LLM 判定，同一测试集）：CE 0.562、LLM 0.734。
**蒸馏目标：≥0.70；失败线：<0.62。**

## 回传

```bash
tar czf model.tar.gz model metrics.json
```

把压缩包放到能访问的位置，或者直接告诉我地址。
