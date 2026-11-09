"""Aggregate existing and extended experiment artifacts into ACL tables."""
import glob,json,os,re
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); RES=os.path.join(ROOT,'results')

def load(p):
 try:
  with open(p,encoding='utf8') as f:return json.load(f)
 except:return None

def main():
 out={'endpoint':[],'coverage':[],'diagnostics':[]}
 for p in sorted(glob.glob(os.path.join(RES,'e13_summary*.json'))):
  x=load(p)
  if not x:continue
  # most e13 summaries have nested mode/all/em
  for mode,v in x.items():
   if isinstance(v,dict) and isinstance(v.get('all'),dict) and 'em' in v['all']:
    out['endpoint'].append({'file':os.path.basename(p),'mode':mode,'n':v['all'].get('n'),'em':v['all'].get('em')})
 for p in sorted(glob.glob(os.path.join(RES,'e23_*.json'))):
  x=load(p)
  if not x or 'summary' not in x:continue
  s=x['summary']; rec={'file':os.path.basename(p),'n':s.get('n'),'k':s.get('k'),'plan':s.get('plan'),'model':s.get('model')}
  for k,v in s.items():
   if any(k.endswith(z) for z in ('_cov','_any','_EM','_F1')): rec[k]=v
  out['coverage'].append(rec)
 for p in ('g9e_diag_musique400.txt','g10_squad200.txt','g10_trivia200.txt'):
  fp=os.path.join(RES,p)
  if os.path.exists(fp): out['diagnostics'].append({'file':p,'text':open(fp).read().strip()})
 with open(os.path.join(RES,'acl_extended_results.json'),'w',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 with open(os.path.join(RES,'acl_extended_results.md'),'w',encoding='utf8') as f:
  f.write('# ACL 扩展实验汇总\n\n## 端到端结果\n\n| 文件 | mode | n | EM |\n|---|---|---:|---:|\n')
  for r in out['endpoint']: f.write(f"| {r['file']} | {r['mode']} | {r['n']} | {r['em']} |\n")
  f.write('\n## 覆盖与预算结果\n\n| 文件 | k | plan | model | seq_cov | seq_EM | cover_cov | cover_EM |\n|---|---:|---|---|---:|---:|---:|---:|\n')
  for r in out['coverage']: f.write(f"| {r['file']} | {r.get('k')} | {r.get('plan')} | {r.get('model')} | {r.get('seq_cov','')} | {r.get('seq_EM','')} | {r.get('cover_cov','')} | {r.get('cover_EM','')} |\n")
  f.write('\n## 其他诊断\n\n')
  for r in out['diagnostics']:f.write(f"### {r['file']}\n\n```text\n{r['text']}\n```\n\n")
 print('wrote results/acl_extended_results.{json,md}')
if __name__=='__main__':main()
