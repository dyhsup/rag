# E11 训练包：逐点蒸馏 + 成对偏好

## 数据

| 文件 | 条数 | 用途 |
|---|---|---|
| `pairs_pointwise.jsonl` | 36,929 | 教师分数 0–10，BCE |
| `pairs_pairwise.jsonl` | 4,450 | gold vs 难负例，margin |

成对数据由 LLM 二选一产生（A/B 随机换位）；8,546 条里 4,096 条被判平局，已丢弃。
非平局中 89.3% 教师选 gold。

## 损失

`BCE(pointwise) + 0.5 · relu(1.0 − (s_gold − s_neg))`

## 运行

```bash
pip install -r requirements.txt
python train_gpu.py --smoke
python train_gpu.py
```

T4 约 2–4 分钟/epoch（含 3× 难例过采样）。

## 评测

按 qid 70/30 分组，只在留出问题上评测；`gap` 档为关键指标。

**目标**：gap closure@3 ≥ 0.30；失败线 < 0.25。
注意：本脚本的候选集是挖掘出的每跳约 10 条，比线上 top-20 口径宽松；最终以回归到项目 E5/E6 口径的复评为准。
