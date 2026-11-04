# E12 三路对照：GT 逐点 / GT 成对 / 教师成对

同一学生、同一训练/测试切分、同一评测，只换监督信号。

## 数据

| 文件 | 条数 | 标签 | 用途 |
|---|---|---|---|
| `pairs_gt_pointwise.jsonl` | 36,929 | `label` 0/1（gold chunk） | `gt_pointwise` |
| `pairs_gt_pairwise.jsonl` | 8,546 | `pref=1`（gold > 难负例） | `gt_pairwise` |
| `pairs_teacher_pairwise.jsonl` | 8,546 | `pref2`（LLM 强制 A/B） | `teacher_pairwise` |

教师成对标签的统计：**gold 胜率 71.0%**（6,069 / 8,546），无 TIE。
GT 成对标签：全部 `pref=1`（按 gold 定义）。

## 运行

```bash
pip install -r requirements.txt

python train_gpu.py --mode gt_pointwise
python train_gpu.py --mode gt_pairwise
python train_gpu.py --mode teacher_pairwise
```

每次约 2–5 分钟（T4）。产物：`model_<mode>/`、`metrics_<mode>.json`。

## 评测

三路共用同一留出集（按 qid 70/30），分档报告 closure@3：

- `all` / `easy(gold<=3)` / **`gap(gold>3)`** / `base_conf_wrong`

`metrics_<mode>.json` 里同时给出 `base` 与 `trained` 的对照。

## 判定

| 结果 | 结论 |
|---|---|
| `gt_pairwise` ≥ `teacher_pairwise` | 放弃 LLM 教师，训练完全用 GT，零 API 依赖 |
| `teacher_pairwise` 明显更好 | 教师成对标签有价值，保留蒸馏 |
| 两者都 < 0.25 | 匹配器不是杠杆，论文指标改为 closure@10 + Recall@20 |

**注意**：本脚本的候选集是挖掘出的每跳约 10 条，比线上 top-20 口径宽松；最终以回归项目 E5/E6 口径的复评为准。
