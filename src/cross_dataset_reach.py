"""Fast cross-dataset candidate reachability under whole-query BM25.
No hop labels are inferred for datasets without them; reports passage-level gold reachability only."""
import argparse,json,os,statistics
from .musique import load
from .retriever import QuestionIndex
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));RES=os.path.join(ROOT,'results')
def run(data,n=200,ks=(1,3,5,10,20,50),tag='cross_reach'):
 qs=load(os.path.join(ROOT,'data','raw',data))[:n]; rows=[]
 for q in qs:
  docs=[{'doc_id':d['doc_id'],'title':d.get('title',''),'text':d['text'],'sentences':d.get('sentences',[d['text']])} for d in q['docs']]
  idx=QuestionIndex({'qid':q['qid'],'docs':docs}); scores=idx.score_docs(q['question']); order=sorted(range(len(docs)),key=lambda i:-scores[i]); gold={i for i,d in enumerate(q['docs']) if d.get('is_gold')}
  rows.append({'qid':q['qid'],'n_gold':len(gold),'reach':{str(k):int(bool(gold&set(order[:k]))) for k in ks},'cov':{str(k):len(gold&set(order[:k]))/max(1,len(gold)) for k in ks}})
 out={'dataset':data,'n':len(rows),'ks':list(ks)}
 for k in ks: out[f'recall_any@{k}']=statistics.mean(r['reach'][str(k)] for r in rows);out[f'coverage@{k}']=statistics.mean(r['cov'][str(k)] for r in rows)
 os.makedirs(RES,exist_ok=True);json.dump({'summary':out,'records':rows},open(os.path.join(RES,tag+'.json'),'w'),ensure_ascii=False,indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--n',type=int,default=200);ap.add_argument('--tag',default='cross_reach');a=ap.parse_args();run(a.data,a.n,tag=a.tag)
