"""Run cached E4 baseline methods on arbitrary MuSiQue-compatible data cache."""
import argparse,json,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from .e4_pipeline import run_vanilla,run_ircot,run_selfask
from .e3_binding import em,f1
from .llm import LLM
from .ce_client import CERemote
from .musique import load
from .retriever import QuestionIndex
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));RES=os.path.join(ROOT,'results')
def main(data,n,methods,tag,workers=8):
 qs=load(os.path.join(ROOT,'data','raw',data))[:n]; docs=[{'doc_id':d['doc_id'],'title':d.get('title',''),'text':d['text'],'sentences':d.get('sentences',[d['text']])} for q in qs for d in q['docs']]; idx=QuestionIndex({'qid':'pool','docs':docs});llm=LLM(os.path.join(ROOT,'cache','llm.sqlite'));ce=CERemote()
 allout={}
 for method in methods:
  path=os.path.join(RES,f'{tag}_{method}.jsonl'); done={}
  if os.path.exists(path): done={json.loads(x)['qid'] for x in open(path) if x.strip()}
  todo=[q for q in qs if q['qid'] not in done]
  def work(q):
   fn={'vanilla':run_vanilla,'ircot':run_ircot,'selfask':run_selfask}[method];pred,info=fn(q,idx,llm,ce);return {'qid':q['qid'],'pred':pred,'gold':q['answer'],'em':em(pred,q['answer']),'f1':f1(pred,q['answer']),'info':info}
  with ThreadPoolExecutor(max_workers=workers) as ex:
   for fut in as_completed([ex.submit(work,q) for q in todo]):
    r=fut.result();open(path,'a',encoding='utf8').write(json.dumps(r,ensure_ascii=False)+'\n')
  rs=[json.loads(x) for x in open(path) if x.strip()];allout[method]={'n':len(rs),'em':sum(r['em'] for r in rs)/len(rs),'f1':sum(r['f1'] for r in rs)/len(rs),'file':'results/'+os.path.basename(path)}
 out=os.path.join(RES,f'{tag}_summary.json');json.dump(allout,open(out,'w'),ensure_ascii=False,indent=2);print(json.dumps(allout,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='musique400.jsonl');ap.add_argument('--n',type=int,default=400);ap.add_argument('--methods',default='vanilla,selfask');ap.add_argument('--tag',default='e4_musique400_extended');a=ap.parse_args();main(a.data,a.n,a.methods.split(','),a.tag)
