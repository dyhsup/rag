# E10 训练包：全量 MuSiQue 难例 + 教师蒸馏

## 规模

| 版本 | 题目 | 训练对 | 难正例 | 难负例 | 自信判错 |
|---|---|---|---|---|---|
| E7 | 200 | 5,400 | 50 | 0 | - |
| E8 | 400 | 4,017 | 649 | 1,538 | 892 |
| E9 | 1,000 | 16,643 | 1,772 | 7,449 | 3,698 |
| **E10** | **2,417** | **36,929** | **4,263** | **18,242** | **8,411** |

教师与初始 CE 一致率 **69.9%**。

## 数据字段

qid, hop, idx, q, p, kind, label, dense_rank, gold_rank, score(教师 0-10), base_logit

## 运行

```bash
pip install -r requirements.txt
python train_gpu.py --smoke
python train_gpu.py
```

T4 约 1.5–3 分钟/epoch（36,929 条 + 3x 难例过采样），总计 6–12 分钟。

## 评测

按 (qid, hop) 分组，只在留出问题（30%，约 725 题）上评测：

- all / easy(gold<=3) / **gap(gold>3)** / base_conf_wrong

**目标**：gap 档 closure@3 >= 0.65；失败线 < 0.55。
