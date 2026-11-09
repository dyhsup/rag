"""E26: structured fact grounding experiment.

The algorithmic object is requirement -> typed fact grounding. A single LLM
call per question extracts candidate subject/relation/object facts from the
20 passages; selection is deterministic and does not use gold answers. The
LLM is an engineering parser, not the matcher or an oracle judge.
"""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor
from .llm import LLM, extract_json
from .musique import load
from .e3_binding import render
from .ce_client import CERemote

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); RES=os.path.join(ROOT,'results')
PROMPT='''Extract explicit atomic facts from each passage. Do not use outside knowledge and do not infer missing facts. Return JSON only as an array of objects, one per passage, each with "passage": integer and "facts": list. Each fact has subject, relation, object, subject_type, object_type, direction. Keep short surface spans. Passages:\n{docs}\nJSON:'''
REL={
 'spouse':{'spouse','married','marriage','husband','wife'},'director':{'director','directed','helmed'},
 'author':{'author','written','wrote'},'actor':{'actor','actress','played','plays','role','character'},
 'location':{'located','location','city','country','state','province'},'education':{'university','college','educated','school','attended','went'},
 'date':{'year','date','born','founded','built','opened','occupied'},'nationality':{'nationality','national','citizen'},
 'award':{'award','won','winner','prize','palme'}}
TYPE={'person':{'who','person','actor','actress','director','author','spouse','husband','wife'},'location':{'where','city','country','state','province','location','university'},'date':{'year','when','date','born','founded','built','opened'},'title':{'film','book','title','movie'}}
TOK=re.compile(r'[a-z0-9]+')

def ts(x): return set(TOK.findall(str(x).lower()))
def fam(x):
 t=ts(x); return {k for k,v in REL.items() if t&v}
def typ(x):
 t=ts(x); return {k for k,v in TYPE.items() if t&v}
def score(req,fact):
 if not isinstance(fact,dict): return 0.
 rt=ts(req); sub=ts(fact.get('subject','')); obj=ts(fact.get('object','')); rel=fam(req)&fam(fact.get('relation',''))
 # Requirement mentions a bound head; reward subject alignment. Do not use gold answer.
 ent=max(len(sub&rt)/max(1,len(sub)), len(obj&rt)/max(1,len(obj)))
 r=1.0 if rel else 0.0
 t=1.0 if typ(req)&({fact.get('subject_type',''),fact.get('object_type','')}) else 0.0
 return .55*ent+.30*r+.15*t

def local_facts(text):
 # Conservative offline fallback: only explicit relation templates; no answer labels.
 out=[]
 for sent in re.split(r'(?<=[.!?])\s+', text):
  s=sent.strip(); low=s.lower()
  patterns=[
   (r'(.+?)\s+as\s+(.+)', 'played'),
   (r'(.+?)\s+was\s+(?:the\s+)?director\s+of\s+(.+)', 'directed_by'),
   (r'(.+?)\s+is\s+(?:the\s+)?author\s+of\s+(.+)', 'author_of'),
   (r'(.+?)\\s+(?:is|was)\\s+located\\s+in\\s+(.+)', 'located_in'),
   (r'(.+?)\\s+(?:married|was married to)\\s+(.+)', 'spouse'),
   (r'(.+?)\\s+(?:attended|went to)\\s+(.+)', 'educated_at'),
  ]
  for pat,rel in patterns:
   m=re.search(pat,s,re.I)
   if m:
    out.append({'subject':m.group(1).strip(),'relation':rel,'object':m.group(2).strip(),'subject_type':'other','object_type':'other','direction':'subject_to_object'}); break
 return out

def parse(text,n,docs=None):
 try: x=extract_json(text,expect='array')
 except Exception: x=None
 if not isinstance(x,list):
  return [{'passage':i,'facts':local_facts(docs[i]) if docs else []} for i in range(n)]
 by={int(z.get('passage')):z.get('facts',[]) for z in x if isinstance(z,dict) and str(z.get('passage','')).isdigit()}
 return [{'passage':i,'facts':by.get(i,[]) or (local_facts(docs[i]) if docs else [])} for i in range(n)]

def metrics(q,selected):
 hop_docs=[h['para_idx'] for h in q['hops']]
 idx={d['idx']:i for i,d in enumerate(q['docs'])}
 hit=[int(idx.get(p,-1) in selected) for p in hop_docs]
 seg=set(idx.get(p,-1) for p in hop_docs)
 return hit, len(set(selected)&seg)/max(1,len(seg)), int(all(hit))

def run(n=20,k=None,workers=4,tag='e26_fact_grounding',model=None):
 qs=load()[:n]; llm=LLM(os.path.join(ROOT,'cache','llm.sqlite'))
 if model:
  os.environ['CE_LOCAL_PATH']=model
 ce=CERemote()
 def one(q):
  docs=[d['title']+'\n'+d['text'] for d in q['docs'][:20]]
  block='\n\n'.join(f'PASSAGE {i}:\n{x}' for i,x in enumerate(docs))
  try:
   obj=llm.json_call(PROMPT.format(docs=block),max_tokens=1800,expect='array')
   raw=json.dumps(obj,ensure_ascii=False)
  except Exception:
   raw='[]'
  facts=parse(raw,len(docs),docs); reqs=[render(h['question'],[x['answer'] for x in q['hops'][:h['hop']]]) for h in q['hops'][:4]]
  per=[]
  for req in reqs:
   vals=[]
   for z in facts: vals.append(max([score(req,f) for f in z['facts']] or [0.]))
   per.append(vals)
  budget=k or len(reqs); sel=[]
  for vals in per:
   if len(sel)>=budget: break
   rest=[i for i in range(20) if i not in sel]; sel.append(max(rest,key=lambda i:vals[i]))
  # CE baseline with identical passage text and same sequential allocation.
  ce_sel=[]
  for req in reqs:
   if len(ce_sel)>=budget: break
   vals=ce.rerank(req,[d['text'] for d in q['docs'][:20]])
   rest=[i for i in range(20) if i not in ce_sel]; ce_sel.append(max(rest,key=lambda i:vals[i]))
  fh,fs,ff=metrics(q,sel); ch,cs,cf=metrics(q,ce_sel)
  return {'qid':q['qid'],'n_hops':len(reqs),'fact_sel':sel,'ce_sel':ce_sel,'fact_hop':fh,'ce_hop':ch,'fact_seg_cov':fs,'ce_seg_cov':cs,'fact_full_hop':ff,'ce_full_hop':cf,'n_facts':sum(len(z['facts']) for z in facts)}
 with ThreadPoolExecutor(max_workers=workers) as ex: rows=list(ex.map(one,qs))
 out={'n':len(rows),'k':k or 'm','plan':'gold','binding':'gold','fact_hop_cov':sum(sum(x['fact_hop'])/len(x['fact_hop']) for x in rows)/len(rows),'ce_hop_cov':sum(sum(x['ce_hop'])/len(x['ce_hop']) for x in rows)/len(rows),'fact_seg_cov':sum(x['fact_seg_cov'] for x in rows)/len(rows),'ce_seg_cov':sum(x['ce_seg_cov'] for x in rows)/len(rows),'fact_full_hop':sum(x['fact_full_hop'] for x in rows)/len(rows),'ce_full_hop':sum(x['ce_full_hop'] for x in rows)/len(rows),'avg_facts':sum(x['n_facts'] for x in rows)/len(rows)}
 os.makedirs(RES,exist_ok=True); json.dump({'summary':out,'records':rows},open(os.path.join(RES,tag+'.json'),'w'),ensure_ascii=False,indent=2); print(json.dumps(out,indent=2)); return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=20); ap.add_argument('--k',type=int,default=None); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--model',default='dist/reqce/model_hop'); ap.add_argument('--tag',default='e26_fact_grounding'); a=ap.parse_args(); run(a.n,a.k,a.workers,a.tag,a.model)
