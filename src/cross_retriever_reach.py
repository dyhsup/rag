"""Passage-level whole-query reachability for BM25, dense, and hybrid."""
import argparse,json,os,statistics
import numpy as np
from .musique import load
from .retriever import QuestionIndex
from .ce_client import CERemote
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));RES=os.path.join(ROOT,'results')
def run(data,n=200,tag='retriever_reach'):
 qs=load(os.path.join(ROOT,'data','raw',data))[:n]; ce=CERemote(); rows=[]
 for q in qs:
  docs=[{'doc_id':d['doc_id'],'title':d.get('title',''),'text':d['text'],'sentences':d.get('sentences',[d['text']])} for d in q['docs']]; texts=[d['text'] for d in docs]; idx=QuestionIndex({'qid':q['qid'],'docs':docs}); bm=np.array(idx.score_docs(q['question']))
  try:
   de=np.array(ce.embed(texts)); qe=np.array(ce.embed([q['question']])[0]); dn=de@qe
  except Exception:
   dn=np.zeros(len(texts))
  gold={i for i,d in enumerate(q['docs']) if d.get('is_gold')}; rows.append({'qid':q['qid'],'gold':len(gold),'scores':{'bm25':bm.tolist(),'dense':dn.tolist()},'gold_idx':sorted(gold)})
 out={'dataset':data,'n':len(rows),'ks':[1,3,5,10,20,50]}
 for method in ('bm25','dense','hybrid'):
  for k in out['ks']:
   vals=[]
   for r in rows:
    bm=np.array(r['scores']['bm25']);dn=np.array(r['scores']['dense']); s=bm if method=='bm25' else dn if method=='dense' else (bm/(np.linalg.norm(bm)+1e-9)+dn/(np.linalg.norm(dn)+1e-9)); order=np.argsort(-s)[:k]; vals.append(len(set(order)&set(r['gold_idx']))/max(1,r['gold']))
   out[f'{method}_coverage@{k}']=statistics.mean(vals)
 os.makedirs(RES,exist_ok=True);json.dump({'summary':out,'records':rows},open(os.path.join(RES,tag+'.json'),'w'),ensure_ascii=False,indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--n',type=int,default=200);ap.add_argument('--tag',default='retriever_reach');a=ap.parse_args();run(a.data,a.n,a.tag)
