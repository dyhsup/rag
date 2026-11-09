"""Stratified paired metrics from frozen E13 query-slot artifacts."""
import json,random,collections,os
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); RES=os.path.join(ROOT,'results')
def rows(p):return [json.loads(x) for x in open(p,encoding='utf8')]
def boot(d,seed=42,n=5000):
 r=random.Random(seed); z=[]
 for _ in range(n): z.append(sum(r.choice(d) for _ in d)/len(d))
 z.sort();return [z[int(.025*n)],z[int(.975*n)]]
def main():
 a={r['qid']:r for r in rows(os.path.join(RES,'e13_select_musique400_gpu.jsonl'))}
 b={r['qid']:r for r in rows(os.path.join(RES,'e13_select_musique400_gpuq.jsonl'))}
 out=[]
 for h in sorted(set(r['n_hops'] for r in a.values())):
  qs=[q for q in a if q in b and a[q]['n_hops']==h]; d=[a[q]['em']-b[q]['em'] for q in qs]
  out.append({'stratum':f'{h}-hop','n':len(d),'bound_em':sum(a[q]['em'] for q in qs)/len(d),'whole_em':sum(b[q]['em'] for q in qs)/len(d),'delta':sum(d)/len(d),'ci':boot(d)})
 d=[a[q]['em']-b[q]['em'] for q in a if q in b];out.append({'stratum':'all','n':len(d),'bound_em':sum(a[q]['em'] for q in a if q in b)/len(d),'whole_em':sum(b[q]['em'] for q in a if q in b)/len(d),'delta':sum(d)/len(d),'ci':boot(d)})
 json.dump(out,open(os.path.join(RES,'extended_queryslot_metrics.json'),'w'),ensure_ascii=False,indent=2)
 with open(os.path.join(RES,'extended_queryslot_metrics.md'),'w') as f:
  f.write('# Extended query-slot stratified metrics\n\n| stratum | n | whole EM | bound EM | delta | 95% CI |\n|---|---:|---:|---:|---:|---|\n')
  for x in out:f.write(f"| {x['stratum']} | {x['n']} | {x['whole_em']:.3f} | {x['bound_em']:.3f} | {x['delta']:.3f} | [{x['ci'][0]:.3f},{x['ci'][1]:.3f}] |\n")
if __name__=='__main__':main()
