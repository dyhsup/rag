# 每数据集 CE 训练包

## 目的

消除主实验里的 reranker 混淆：给每个数据集训练一个 CE，训练/测试按问题切分，评测集与训练集不重叠。

## 数据

| 文件 | 对数 | 问题数 | 来源 |
|---|---|---|---|
| `musique.jsonl` | 12,000 | 1,664 | MuSiQue，已排除评测的 200 题 |
| `squad.jsonl` | 6,000 | 600 | SQuAD train |
| `trivia.jsonl` | 6,769 | 600 | TriviaQA rc.web train |
| `wiki2.jsonl` | 5,770 | 600 | 2Wiki train |
| `hotpot.jsonl` | 5,988 | 600 | HotpotQA train |

字段：`qid, q, p, label`（1=gold，0=负例）；每题的池子为 1–3 正 + 9 负。

## 训练

```bash
pip install -r requirements.txt

python train_ce.py --data musique.jsonl --out models/musique
python train_ce.py --data squad.jsonl   --out models/squad
python train_ce.py --data trivia.jsonl  --out models/trivia
python train_ce.py --data wiki2.jsonl   --out models/wiki2
python train_ce.py --data hotpot.jsonl  --out models/hotpot
```

- 默认 3 epochs、lr 2e-5、batch 32、max_len 256。
- 按 `qid` 留 10% 做 AUC 校验；模型只在训练切分上训练。
- T4 上每个数据集约 **1–3 分钟**。
- 输出 `models/<dataset>/`，可直接 `from_pretrained`。

## 训练完怎么用

把 `models/<dataset>` 作为 CE 服务加载（接口与现有服务一致：`/rerank`、`/score`），
然后所有方法（vanilla / IRCoT / RBG）使用**同一个**该数据集的 CE 重跑评测。

## 判定

- 如果 `RBG − IRCoT(+CE_domain)` 仍显著为正 → 流水线贡献成立；
- 如果差距消失 → 论文应改为诊断/基准定位，不再主张端到端增益。
