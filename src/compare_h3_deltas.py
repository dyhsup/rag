import json,os,glob
from eval import bootstrap_ci,paired_bootstrap
ROOT='.';RES='results'
def read(p):return {json.loads(x)['qid']:json.loads(x) for x in open(p) if x.strip()}
sets=[('musique200','h3_musique200','e4_musique_ircot.jsonl','e4_musique_vanilla.jsonl'),('musique400','h3_musique400','e4_musique400_ircot.jsonl',None),('wiki2','h3_wiki2200','e4_wiki2_200_ircot.jsonl',None),('hotpot','h3_hotpot200','e4_hotpot_ircot.jsonl','e4_hotpot_vanilla.jsonl'),('mhr','h3_mhr200','e4_mhr_200_ircot.jsonl',None)]
out=[]
for ds,tag,oldirc,oldvan in sets:
 for m in ('vanilla','ircot','selfask'):
  p=f'{RES}/{tag}_{m}.jsonl';
  if not os.path.exists(p):continue
  nr=read(p); row={'dataset':ds,'method':m,'n':len(nr),'new_em':sum(x['em'] for x in nr.values())/len(nr),'new_f1':sum(x['f1'] for x in nr.values())/len(nr),'new_em_ci':bootstrap_ci([x['em'] for x in nr.values()])}
  old=None
  if m=='ircot':old=oldirc
  if m=='vanilla' and oldvan:old=oldvan
  if old and os.path.exists(f'{RES}/{old}'):
   od=read(f'{RES}/{old}');keys=sorted(set(nr)&set(od));d=paired_bootstrap([nr[k]['em'] for k in keys],[od[k]['em'] for k in keys]);row.update({'old_file':'results/'+old,'old_em':sum(od[k]['em'] for k in keys)/len(keys),'paired_delta_new_minus_old':d[0],'paired_ci':d[1:]})
  out.append(row)
json.dump(out,open(f'{RES}/h3_difference_analysis.json','w'),ensure_ascii=False,indent=2)
with open(f'{RES}/h3_difference_analysis.md','w') as f:
 f.write('# H3/Qwen3.8 与旧结果差异分析\n\n|数据集|方法|n|新EM|旧EM|新-旧配对ΔEM|95% CI|\n|---|---|---:|---:|---:|---:|---|\n')
 for r in out:
  old='—' if 'old_em' not in r else f"{r['old_em']:.3f}"
  delta='—' if 'paired_delta_new_minus_old' not in r else f"{r['paired_delta_new_minus_old']:+.3f}"
  f.write(f"|{r['dataset']}|{r['method']}|{r['n']}|{r['new_em']:.3f}|{old}|{delta}|{r.get('paired_ci','—')}|\n")
print(json.dumps(out,indent=2))
