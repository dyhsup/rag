import json,os,statistics
from eval import bootstrap_ci,paired_bootstrap
RES='results'
def read(p):return {json.loads(x)['qid']:json.loads(x) for x in open(p) if x.strip()}
sets={'musique200':'h3e2e_musique200','wiki2':'h3e2e_wiki2200','hotpot':'h3e2e_hotpot200','mhr':'h3e2e_mhr200'}
rows=[]
for ds,t in sets.items():
 for m in ['rewrite','hyde','decomp']:
  r=read(f'{RES}/{t}_{m}.jsonl');ems=[x['em'] for x in r.values()];f1s=[x['f1'] for x in r.values()];rows.append({'dataset':ds,'method':m,'n':len(r),'em':sum(ems)/len(ems),'f1':sum(f1s)/len(f1s),'em_ci':bootstrap_ci(ems),'f1_ci':bootstrap_ci(f1s)})
# compare new extra methods against H3 IRCoT/vanilla where available
for ds,t in [('musique200','h3_musique200'),('wiki2','h3_wiki2200'),('hotpot','h3_hotpot200'),('mhr','h3_mhr200')]:
 base=read(f'{RES}/{t}_ircot.jsonl')
 for m in ['rewrite','hyde','decomp']:
  cur=read(f'{RES}/{sets[ds]}_{m}.jsonl');keys=sorted(set(base)&set(cur));d=paired_bootstrap([cur[k]['em'] for k in keys],[base[k]['em'] for k in keys]);rows.append({'dataset':ds,'method':m+'-vs-H3-IRCoT','n':len(keys),'delta_em':d[0],'ci':d[1:]})
json.dump(rows,open(f'{RES}/robustness_metrics.json','w'),ensure_ascii=False,indent=2)
with open(f'{RES}/robustness_metrics.md','w') as f:
 f.write('# 扩展端到端稳健性指标\n\n|数据集|方法|n|EM|EM 95% CI|F1|\n|---|---|---:|---:|---|---:|\n')
 for r in rows:
  if 'em' in r:f.write(f"|{r['dataset']}|{r['method']}|{r['n']}|{r['em']:.3f}|{r['em_ci']}|{r['f1']:.3f}|\n")
 f.write('\n## 同模型配对差异（相对 H3 IRCoT）\n\n|数据集|方法|ΔEM|95% CI|\n|---|---|---:|---|\n')
 for r in rows:
  if 'delta_em' in r:f.write(f"|{r['dataset']}|{r['method']}|{r['delta_em']:+.3f}|{r['ci']}|\n")
