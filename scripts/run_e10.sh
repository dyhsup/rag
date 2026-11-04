#!/bin/bash
set -e
cd /disk1/query-evidance
mkdir -p logs dist/e10
rm -f logs/e10_DONE logs/e10_FAIL

curl -s --noproxy '*' -m 10 http://10.52.100.210:8833/health > logs/e10_health.json || { echo "CE service down"; touch logs/e10_FAIL; exit 1; }

python3 -m src.e8_mine --n 2417 --hard-pos 5 --hard-neg 3 --easy-neg 1 \
  --data musique2417.jsonl > logs/e10_mine.log 2>&1

python3 -m src.e8_label --workers 12 > logs/e10_label.log 2>&1

python3 - > logs/e10_post.log 2>&1 <<'PY'
import json, math, collections, os
from src.ce_client import CERemote
RES='results'
recs=[json.loads(l) for l in open(f'{RES}/e8_labeled.jsonl')]
seen={}
for r in recs:
    seen[(r['qid'],r['hop'],r['idx'])]=r
u=list(seen.values())
ce=CERemote()
for i in range(0,len(u),512):
    part=u[i:i+512]
    s=ce.score([(r['q'],r['p']) for r in part])
    for r,x in zip(part,s): r['base_logit']=x
with open(f'{RES}/e10_scored.jsonl','w') as f:
    for r in u: f.write(json.dumps(r,ensure_ascii=False)+'\n')
os.makedirs('dist/e10',exist_ok=True)
with open('dist/e10/pairs.jsonl','w') as f:
    for r in u: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def sig(x): return 1/(1+math.exp(-x))
agree=sum(1 for r in u if (r['score']>=8)==(sig(r['base_logit'])>=0.5))
hw=sum(1 for r in u if (r['score']>=8 and sig(r['base_logit'])<0.2) or (r['score']<8 and sig(r['base_logit'])>0.8))
c=collections.Counter(r['kind'] for r in u)
stats={'unique':len(u),'kinds':dict(c),'agreement':agree/len(u),'conf_wrong':hw}
print(json.dumps(stats,ensure_ascii=False,indent=2))
open('results/e10_stats.json','w').write(json.dumps(stats,ensure_ascii=False,indent=2))

readme=f"""# E10 训练包：全量 MuSiQue 难例 + 教师蒸馏

## 规模

| 版本 | 题目 | 训练对 | 难正例 | 难负例 | 自信判错 |
|---|---|---|---|---|---|
| E7 | 200 | 5,400 | 50 | 0 | - |
| E8 | 400 | 4,017 | 649 | 1,538 | 892 |
| E9 | 1,000 | 16,643 | 1,772 | 7,449 | 3,698 |
| **E10** | **2,417** | **{len(u):,}** | **{c.get('hard_pos',0):,}** | **{c.get('hard_neg',0):,}** | **{hw:,}** |

教师与初始 CE 一致率 **{agree/len(u):.1%}**。

## 数据字段

qid, hop, idx, q, p, kind, label, dense_rank, gold_rank, score(教师 0-10), base_logit

## 运行

```bash
pip install -r requirements.txt
python train_gpu.py --smoke
python train_gpu.py
```

T4 约 1.5–3 分钟/epoch（{len(u):,} 条 + 3x 难例过采样），总计 6–12 分钟。

## 评测

按 (qid, hop) 分组，只在留出问题（30%，约 725 题）上评测：

- all / easy(gold<=3) / **gap(gold>3)** / base_conf_wrong

**目标**：gap 档 closure@3 >= 0.65；失败线 < 0.55。
"""
open('dist/e10/README.md','w').write(readme)
PY

cd dist && rm -rf e10/__pycache__ && tar czf e10-bundle.tar.gz e10/
touch logs/e10_DONE
