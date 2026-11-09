"""Build a unified comparable-method table from frozen result artifacts.
Only rows with explicit dataset/pool/reader/CE metadata are grouped together.
"""
import json,os,glob
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));RES=os.path.join(ROOT,'results')
SPECS=[
 ('MuSiQue-200','Vanilla','e4_musique_vanilla.jsonl','zero-shot CE/BM25/standard reader'),
 ('MuSiQue-200','IRCoT','e4_musique_ircot.jsonl','zero-shot CE/BM25/IRCoT reader'),
 ('MuSiQue-200','Self-Ask','e4_musique_selfask.jsonl','zero-shot CE/BM25/Self-Ask'),
 ('MuSiQue-200','HyDE','g4_hyde.jsonl','zero-shot CE/BM25/HyDE'),
 ('MuSiQue-200','CovBind','e13_select_musique200_gpu.jsonl','domain CE/bound query/CoT reader'),
 ('MuSiQue-200','CovBind-whole-query','e13_select_musique200_gpuq.jsonl','domain CE/whole query/CoT reader'),
 ('MuSiQue-400','Vanilla','e4_musique400_extended_vanilla.jsonl','zero-shot CE/BM25/standard reader'),
 ('MuSiQue-400','Self-Ask','e4_musique400_extended_selfask.jsonl','zero-shot CE/BM25/Self-Ask'),
 ('MuSiQue-400','IRCoT','e4_musique400_ircot.jsonl','zero-shot CE/BM25/IRCoT reader'),
 ('MuSiQue-400','CovBind','e13_select_musique400_gpu.jsonl','domain CE/bound query/CoT reader'),
 ('MuSiQue-400','CovBind-whole-query','e13_select_musique400_gpuq.jsonl','domain CE/whole query/CoT reader'),
 ('2Wiki-200','Vanilla','e4_wiki2_extended_vanilla.jsonl','zero-shot CE/BM25/standard reader'),
 ('2Wiki-200','Self-Ask','e4_wiki2_extended_selfask.jsonl','zero-shot CE/BM25/Self-Ask'),
 ('2Wiki-200','IRCoT','e4_wiki2_200_ircot.jsonl','zero-shot CE/BM25/IRCoT reader'),
 ('2Wiki-200','CovBind','e13_select_wiki2_200_cot.jsonl','domain CE/bound query/CoT reader'),
 ('Hotpot-200','Vanilla','e4_hotpot_vanilla.jsonl','zero-shot CE/BM25/standard reader'),
 ('Hotpot-200','Self-Ask','e4_hotpot_extended_selfask.jsonl','zero-shot CE/BM25/Self-Ask'),
 ('Hotpot-200','IRCoT','e4_hotpot_ircot.jsonl','zero-shot CE/BM25/IRCoT reader'),
 ('Hotpot-200','CovBind','e13_select_hotpot200_cot.jsonl','domain CE/bound query/CoT reader'),
 ('MHR-200','Vanilla','e4_mhr_extended_vanilla.jsonl','zero-shot CE/BM25/standard reader'),
 ('MHR-200','Self-Ask','e4_mhr_extended_selfask.jsonl','zero-shot CE/BM25/Self-Ask'),
 ('MHR-200','IRCoT','e4_mhr_200_ircot.jsonl','zero-shot CE/BM25/IRCoT reader'),
]
def main():
 rows=[]
 for ds,m,f,protocol in SPECS:
  p=os.path.join(RES,f)
  if not os.path.exists(p):continue
  rs=[json.loads(x) for x in open(p,encoding='utf8') if x.strip()]
  if not rs:continue
  em=sum(r.get('em',0) for r in rs)/len(rs); f1=sum(r.get('f1',0) for r in rs)/len(rs)
  rows.append({'dataset':ds,'method':m,'n':len(rs),'em':em,'f1':f1,'protocol':protocol,'artifact':'results/'+f})
 json.dump(rows,open(os.path.join(RES,'method_comparison.json'),'w'),ensure_ascii=False,indent=2)
 with open(os.path.join(RES,'method_comparison.md'),'w') as out:
  out.write('# Unified method comparison\n\n> Rows are grouped by dataset but not pooled across protocols. Domain-CE CovBind and zero-shot baseline rows are explicitly labeled.\n\n| dataset | method | n | EM | F1 | protocol |\n|---|---|---:|---:|---:|---|\n')
  for r in rows:out.write(f"| {r['dataset']} | {r['method']} | {r['n']} | {r['em']:.3f} | {r['f1']:.3f} | {r['protocol']} |\n")
 print('rows',len(rows))
if __name__=='__main__':main()
