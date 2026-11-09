import argparse,json,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from .llm import LLM
from .musique import load as load_m
from .data import load_questions
from .retriever import QuestionIndex
from .e3_binding import em,f1
from .e4_pipeline import run_vanilla,run_ircot,run_selfask
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));RES=os.path.join(ROOT,'results')
def load_data(data):
 p=os.path.join(ROOT,'data','raw',data);return load_questions(p) if data.startswith('hotpot') else load_m(p)
def run(data,n,methods,tag,workers=16):
 qs=load_data(data)[:n];docs=[{'doc_id':d['doc_id'],'title':d.get('title',''),'text':d['text'],'sentences':d.get('sentences',[d['text']])} for q in qs for d in q['docs']];idx=QuestionIndex({'qid':'pool','docs':docs});llm=LLM(os.path.join(ROOT,'cache','llm_h3.sqlite'));out={}
 for m in methods:
  path=os.path.join(RES,f'{tag}_{m}.jsonl');done={json.loads(x)['qid'] for x in open(path) if x.strip()} if os.path.exists(path) else set();todo=[q for q in qs if q['qid'] not in done]
  def one(q):
   fn={'vanilla':run_vanilla,'ircot':run_ircot,'selfask':run_selfask}[m];pred,info=fn(q,idx,llm,None);return {'qid':q['qid'],'pred':pred,'gold':q['answer'],'em':em(pred,q['answer']),'f1':f1(pred,q['answer']),'info':info}
  with ThreadPoolExecutor(max_workers=workers) as ex:
   for fut in as_completed([ex.submit(one,q) for q in todo]):open(path,'a',encoding='utf8').write(json.dumps(fut.result(),ensure_ascii=False)+'\n')
  rs=[json.loads(x) for x in open(path) if x.strip()];out[m]={'n':len(rs),'em':sum(x['em'] for x in rs)/len(rs),'f1':sum(x['f1'] for x in rs)/len(rs),'file':'results/'+os.path.basename(path)}
 json.dump(out,open(os.path.join(RES,f'{tag}_summary.json'),'w'),ensure_ascii=False,indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--n',type=int,default=200);ap.add_argument('--methods',default='vanilla,ircot,selfask');ap.add_argument('--tag',default='h3_baseline');a=ap.parse_args();run(a.data,a.n,a.methods.split(','),a.tag)
