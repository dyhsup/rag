# E8 训练包：难例挖掘 + 教师蒸馏

## 相对 E7 的改动

| | E7 | E8 |
|---|---|---|
| 题目数 | 200 | 400 |
| 候选 | dense top-10 | dense top-50 |
| 难正例 | 50（0.9%） | **649（16.2%）** |
| 难负例 | — | 1538 |
| 训练 | 3 epochs | 4 epochs + **难例 3× 过采样** |

## 数据

`pairs.jsonl` 4017 条，字段：

```
qid, hop, idx, q, p, kind, label, dense_rank, gold_rank, score(教师 0-10), base_logit(蒸馏前 ms-marco)
```

- `label=1` 表示 gold chunk
- `base_logit` 用于识别"初始模型自信判错"的 892 条样本（过采样目标）
- 教师与初始 CE 一致率 69.8%

## 运行

```bash
pip install -r requirements.txt
python train_gpu.py --smoke
python train_gpu.py
```

预期：T4 约 20–40 s/epoch（4017 条 + 过采样），总计 1–3 分钟。

## 评测口径

脚本按 `(qid, hop)` 分组，只在**留出问题**上评测，分三档报告 closure@3：

| 档位 | 含义 |
|---|---|
| `all` | 所有 gold 在候选内的 hop |
| `easy(gold<=3)` | 初始检索已经排对的 |
| `gap(gold>3)` | **语义 gap 子集（关键）** |
| `base_conf_wrong` | 初始 CE 自信判错的 hop |

**目标**：`gap` 档从 0.50 提到 ≥0.65。**失败线：<0.55。**
