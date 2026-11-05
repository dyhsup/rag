"""Diagnostics: retrieval recall and answer-support of two readers."""
import argparse, json, os, re
from .ce_client import CERemote
from .e3_binding import em
from .e4_pipeline import topk
from .retriever import QuestionIndex

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); RES=os.path.join(ROOT,'results')
def norm(s):
    s=s.lower(); s=re.sub(r'[^a-z0-9 ]',' ',s); return ' '.join(s.split())

def main(data='squad_200.jsonl', n=200):
    qs=[json.loads(l) for l in open(os.path.join(ROOT,'data','raw',data))][:n]
    docs=[{'doc_id':d['doc_id'],'title':d['title'],'text':d['text'],'sentences':d['sentences']} for q in qs for d in q['docs']]
    idx=QuestionIndex({'qid':'pool','docs':docs}); ce=CERemote(); tag=data.replace('.jsonl','')
    def load(p): return {json.loads(l)['qid']:json.loads(l) for l in open(p)} if os.path.exists(p) else {}
    a=load(os.path.join(RES,f'g9_{tag}_concat.jsonl')); b=load(os.path.join(RES,f'g9d_{tag}_cot5.jsonl'))
    r5=r1=r1_ce=0; sel=[]
    for q in qs:
        cand,_=topk(idx,q['question'],50)
        try:
            sc=ce.rerank(q['question'],cand); order=sorted(range(len(cand)),key=lambda i:-sc[i])
        except Exception:
            order=list(range(len(cand)))
        top=[cand[i] for i in order[:5]]
        gold=[d['text'] for d in q['docs'] if d['is_gold']]
        r5+=int(any(g in top for g in gold)); r1+=int(gold and gold[0]==top[0])
        r1_ce+=int(gold and gold[0] in top)
        pa=norm(a[q['qid']]['pred']) if q['qid'] in a else ''
        pb=norm(b[q['qid']]['pred']) if q['qid'] in b else ''
        sa=any(pa and pa in norm(p) for p in top); sb=any(pb and pb in norm(p) for p in top)
        if sb and not sa: pred=b[q['qid']]['pred']
        elif sa and not sb: pred=a[q['qid']]['pred']
        elif sb and sa: pred=b[q['qid']]['pred']
        else: pred=b[q['qid']]['pred'] if q['qid'] in b else (a[q['qid']]['pred'] if q['qid'] in a else '')
        sel.append(em(pred,q['answer']))
    print(f'{tag}: recall@5={r5/len(qs):.3f} gold@1={r1/len(qs):.3f} gold-in-top5-at-1={r1_ce/len(qs):.3f}  selector EM={sum(sel)/len(sel):.3f}')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='squad_200.jsonl'); a=ap.parse_args(); main(a.data)
