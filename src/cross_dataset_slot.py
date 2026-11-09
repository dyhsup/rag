"""Cross-dataset query-slot/reachability evaluation for MuSiQue-compatible caches."""
import argparse,json,os,statistics,random
from .musique import load
from .e3_binding import render
from .retriever import QuestionIndex
from .ce_client import CERemote
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); RES=os.path.join(ROOT,'results')
def boot(d,seed=42,n=3000):
 r=random.Random(seed); z=[]
 for _ in range(n):z.append(sum(r.choice(d) for _ in d)/len(d))
 z.sort();return [z[int(.025*n)],z[int(.975*n)]]
def run(data='wiki2_200.jsonl',n=200,k=2,topk=50,model=None,tag='cross_slot'):
 qs=load(os.path.join(ROOT,'data','raw',data))[:n]
 if model: os.environ['CE_LOCAL_PATH']=model
 ce=CERemote(); rows=[]
 for q in qs:
  idx=QuestionIndex({'qid':q['qid'],'docs':[{'doc_id':d['doc_id'],'title':d['title'],'text':d['text'],'sentences':d['sentences']} for d in q['docs']]})
  docs=[d['text'] for d in q['docs']]; gold_by_hop={h['hop']:h for h in q.get('hops',[])}; binds=[]; whole=q['question']; whole_cand=idx.score_docs(whole)
  if not gold_by_hop:
   continue
  wo=sorted(range(len(docs)),key=lambda i:-whole_cand[i])[:topk]
  seq=[]; ranks=[]; hoprows=[]
  for h in sorted(gold_by_hop):
   step=gold_by_hop[h]; text=render(step['question'],binds); sc=idx.score_docs(text); order=sorted(range(len(docs)),key=lambda i:-sc[i]); ranks.append(order.index(next((i for i,d in enumerate(q['docs']) if d.get('idx',i)==step['para_idx']),-1))+1 if step['para_idx']>=0 and any(d.get('idx',i)==step['para_idx'] for i,d in enumerate(q['docs'])) else -1)
   cand=order[:topk]; vals=ce.rerank(text,[docs[i] for i in cand]); pick=cand[max(range(len(cand)),key=lambda j:vals[j])] if cand else -1; seq.append(pick); binds.append(step['answer']); hoprows.append({'hop':h,'rank':ranks[-1],'bound_query':text})
  gold=set(i for i,d in enumerate(q['docs']) if d.get('is_gold')); hop_gold=[next((i for i,d in enumerate(q['docs']) if d.get('idx',i)==h['para_idx']),-1) for h in q.get('hops',[])]
  whole_sel=wo[:k]; seq_sel=seq[:k]
  wh_cov=len(gold&set(whole_sel))/max(1,len(gold)); sq_cov=len(gold&set(seq_sel))/max(1,len(gold))
  wh_h=[int(x in whole_sel) for x in hop_gold]; sq_h=[int(x in seq_sel) for x in hop_gold]
  rows.append({'qid':q['qid'],'n_hops':q['n_hops'],'whole_cov':wh_cov,'seq_cov':sq_cov,'whole_hop':wh_h,'seq_hop':sq_h,'whole_full':int(all(wh_h)),'seq_full':int(all(sq_h)),'hop_ranks':ranks,'hoprows':hoprows})
 d=[r['seq_cov']-r['whole_cov'] for r in rows]
 out={'dataset':data,'n':len(rows),'k':k,'topk':topk,'model':model or 'remote','whole_cov':statistics.mean(r['whole_cov'] for r in rows),'seq_cov':statistics.mean(r['seq_cov'] for r in rows),'delta_cov':statistics.mean(d),'delta_cov_ci':boot(d),'whole_full':statistics.mean(r['whole_full'] for r in rows),'seq_full':statistics.mean(r['seq_full'] for r in rows),'hop_recall_whole':statistics.mean(sum(r['whole_hop'])/len(r['whole_hop']) for r in rows),'hop_recall_seq':statistics.mean(sum(r['seq_hop'])/len(r['seq_hop']) for r in rows),'mean_hop_rank_bound':statistics.mean([x for r in rows for x in r['hop_ranks'] if x>0])}
 os.makedirs(RES,exist_ok=True);json.dump({'summary':out,'records':rows},open(os.path.join(RES,tag+'.json'),'w'),ensure_ascii=False,indent=2);print(json.dumps(out,indent=2));return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='wiki2_200.jsonl');ap.add_argument('--n',type=int,default=200);ap.add_argument('--k',type=int,default=2);ap.add_argument('--topk',type=int,default=50);ap.add_argument('--model',default='models/ce_5dataset/models/wiki2');ap.add_argument('--tag',default='cross_slot_wiki2');a=ap.parse_args();run(a.data,a.n,a.k,a.topk,a.model,a.tag)
