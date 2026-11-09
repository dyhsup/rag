"""Additional end-to-end baselines: query rewrite, HyDE, and decomposition retrieval.
All methods use the same per-dataset BM25 pool and the same reader; results are
labeled exploratory when LLM pseudo/rewrite generation is involved.
"""
import argparse,json,os,re
from concurrent.futures import ThreadPoolExecutor,as_completed
from .llm import LLM
from .musique import load as load_m
from .data import load_questions
from .retriever import QuestionIndex
from .e3_binding import em,f1
from .e4_pipeline import fmt,READ,topk
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));RES=os.path.join(ROOT,'results')
REWRITE='''Rewrite the question into one concise search query that preserves all entities and constraints. Output only the query. Question: {q}'''
HYDE='''Write a short factual Wikipedia-style passage that would answer this question. Output only the passage, no caveats. Question: {q}'''
DECOMP='''Split the question into 2-4 independent search queries, each answerable from one passage. Output JSON only: {{"queries":["..."]}}. Question: {q}'''
def load_data(data):
 p=os.path.join(ROOT,'data','raw',data)
 if data.startswith('hotpot'):return load_questions(p)
 return load_m(p)
def run(data,n,methods,tag,workers=8):
 qs=load_data(data)[:n];docs=[{'doc_id':d['doc_id'],'title':d.get('title',''),'text':d['text'],'sentences':d.get('sentences',[d['text']])} for q in qs for d in q['docs']];idx=QuestionIndex({'qid':'pool','docs':docs});llm=LLM(os.path.join(ROOT,'cache','llm.sqlite'))
 def one(q,m):
  try:
   if m=='rewrite': query=llm.chat([{'role':'user','content':REWRITE.format(q=q['question'])}],max_tokens=96).strip().split('\n')[0]; ps,_=topk(idx,query,5);info={'query':query}
   elif m=='hyde': query=llm.chat([{'role':'user','content':HYDE.format(q=q['question'])}],max_tokens=180).strip();ps,_=topk(idx,query,5);info={'pseudo':query}
   elif m=='decomp':
    o=llm.json_call(DECOMP.format(q=q['question']),max_tokens=256);queries=[x for x in o.get('queries',[]) if x][:4];acc=[]
    for x in queries:acc.extend(topk(idx,x,3)[0])
    ps=list(dict.fromkeys(acc))[:8];info={'queries':queries}
   else:ps,_=topk(idx,q['question'],5);info={}
   pred=llm.chat([{'role':'user','content':READ.format(q=q['question'],ps=fmt(ps))}],max_tokens=32).strip().split('\n')[0]
  except Exception as e:
   pred='';info={'error':repr(e)}
  return {'qid':q['qid'],'pred':pred,'gold':q['answer'],'em':em(pred,q['answer']),'f1':f1(pred,q['answer']),'info':info}
 out={}
 for m in methods:
  path=os.path.join(RES,f'{tag}_{m}.jsonl');done={json.loads(x)['qid'] for x in open(path) if x.strip()} if os.path.exists(path) else set();todo=[q for q in qs if q['qid'] not in done]
  with ThreadPoolExecutor(max_workers=workers) as ex:
   for fut in as_completed([ex.submit(one,q,m) for q in todo]):open(path,'a',encoding='utf8').write(json.dumps(fut.result(),ensure_ascii=False)+'\n')
  rs=[json.loads(x) for x in open(path) if x.strip()];out[m]={'n':len(rs),'em':sum(x['em'] for x in rs)/len(rs),'f1':sum(x['f1'] for x in rs)/len(rs),'file':'results/'+os.path.basename(path)}
 json.dump(out,open(os.path.join(RES,f'{tag}_summary.json'),'w'),ensure_ascii=False,indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--n',type=int,default=50);ap.add_argument('--methods',default='rewrite,hyde,decomp');ap.add_argument('--tag',default='e2e_extra');a=ap.parse_args();run(a.data,a.n,a.methods.split(','),a.tag)
