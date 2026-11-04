# E9 训练包：1000 题难例 + 教师蒸馏

## 规模（相对 E8）

| | E7 | E8 | **E9** |
|---|---|---|---|
| 题目 | 200 | 400 | **1000** |
| 跳数 | 540 | 1080 | **2662** |
| 候选 | top-10 | top-50 | top-50 |
| 训练对 | 5,400 | 4,017 | **16,643** |
| 难正例（gold 4–50） | 50 | 649 | **1,772** |
| 难负例 | — | 1,538 | **7,449** |
| 初始 CE 自信判错 | — | 892 | **3,698** |

## 数据字段

```
qid, hop, idx, q, p, kind, label, dense_rank, gold_rank,
score(教师 0-10), base_logit(蒸馏前 ms-marco)
```

- `kind`: hard_pos / easy_pos / hard_neg / easy_neg
- `label=1` 仅表示 gold chunk；训练目标是 `score`
- 教师与初始 CE 一致率 **69.7%**

## 运行

```bash
pip install -r requirements.txt
python train_gpu.py --smoke
python train_gpu.py
```

T4 约 40–80 s/epoch（16.6k 条 + 3× 难例过采样），总计 3–6 分钟。

## 评测口径（脚本自动输出）

按 `(qid, hop)` 分组，只在留出问题（30%，约 300 题）上评测：

| 档位 | 含义 |
|---|---|
| `all` | gold 在候选内的 hop |
| `easy(gold<=3)` | 初始检索已排对 |
| **`gap(gold>3)`** | **语义 gap 子集（关键）** |
| `base_conf_wrong` | 初始 CE 自信判错的 hop |

**目标**：`gap` 档 closure@3 ≥ 0.65；**失败线 < 0.55**。
上一版（E7，200 题）在干净留出集上是 0.50（n=14，噪声大），本版留出 gap 样本预计 200+，数字才可信。
