# ACL 扩展实验汇总

## 端到端结果

| 文件 | mode | n | EM |
|---|---|---:|---:|
| e13_summary.json | ctxlast | 200 | 0.425 |
| e13_summary_hotpot200.json | select | 200 | 0.675 |
| e13_summary_hotpot200_cot.json | select | 200 | 0.68 |
| e13_summary_hotpot200_hybridcot.json | select | 200 | 0.665 |
| e13_summary_mhr_200.json | select | 200 | 0.53 |
| e13_summary_musique200.json | select | 200 | 0.43 |
| e13_summary_musique200_cot.json | select | 200 | 0.43 |
| e13_summary_musique200_gpu.json | select | 200 | 0.44 |
| e13_summary_musique200_gpuq.json | select | 200 | 0.17 |
| e13_summary_musique200_hybridcot.json | select | 200 | 0.44 |
| e13_summary_musique200_qslotq.json | select | 200 | 0.215 |
| e13_summary_musique200_rq.json | select | 200 | 0.45 |
| e13_summary_musique400.json | select | 400 | 0.4275 |
| e13_summary_musique400_gpu.json | select | 400 | 0.425 |
| e13_summary_musique400_gpuq.json | select | 400 | 0.17 |
| e13_summary_musique400_hybridcot.json | select | 400 | 0.435 |
| e13_summary_squad_200.json | select | 200 | 0.585 |
| e13_summary_trivia_200.json | select | 200 | 0.75 |
| e13_summary_wiki2_200.json | select | 200 | 0.595 |
| e13_summary_wiki2_200_cot.json | select | 200 | 0.62 |
| e13_summary_wiki2_200_hybridcot.json | select | 200 | 0.59 |

## 覆盖与预算结果

| 文件 | k | plan | model | seq_cov | seq_EM | cover_cov | cover_EM |
|---|---:|---|---|---:|---:|---:|---:|
| e23_aggreg.json | 2 | gold | /root/reqce/model_hop | 0.7375000000000003 |  | 0.7425000000000002 |  |
| e23_hop_gold.json | 2 | gold | dist/reqce/model_hop | 0.7308333333333334 |  |  |  |
| e23_hop_gold_cover.json | 2 | gold | dist/reqce/model_hop | 0.7308333333333334 |  | 0.7099999999999999 |  |
| e23_hop_gold_read.json | 2 | gold | /root/reqce/model_hop | 0.7375000000000003 | 0.35 |  |  |
| e23_q_gold.json | 2 | gold | dist/reqce/model_q | 0.7125 |  |  |  |
| e23_q_gold_read.json | 2 | gold | /root/reqce/model_q | 0.6841666666666667 | 0.35 |  |  |
| e23_support.json | 2 | gold | /root/reqce/model_support | 0.6633333333333333 | 0.31 |  |  |

## 问题 CE 对照

问题查询训练模型 `dist/reqce/model_q` 在 gold plan、k=2、100 题上的覆盖结果：whole=.5308，seq=.7125，global=.6808，cover=.7117；需求 CE `model_hop` 的对应 seq=.7308。该对照支持“需求查询训练带来额外但有限的覆盖收益”，主要增益仍来自推理时 query slot 替换。

## 其他诊断

### g9e_diag_musique400.txt

```text
musique400: recall@5=0.865 gold@1=0.325 gold-in-top5-at-1=0.515  selector EM=0.000
```

### g10_squad200.txt

```text
[g10] squad_200 vanilla: 0
[g10] squad_200 cot: 0
[g10] squad_200 ircot: 0
```

### g10_trivia200.txt

```text
[g10] trivia_200 vanilla: 0
[g10] trivia_200 cot: 0
[g10] trivia_200 ircot: 0
```

