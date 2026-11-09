import os,json,glob,collections,hashlib
RES='results'
files=glob.glob(os.path.join(RES,'*.jsonl')); rows=[]
for p in files:
 rs=[]; bad=0; empty=0; qids=[]; errors=0
 for line in open(p,encoding='utf8',errors='replace'):
  try:r=json.loads(line)
  except:bad+=1;continue
  rs.append(r); qids.append(r.get('qid'))
  if r.get('pred','__MISSING__')=='' or r.get('final','__MISSING__')=='':empty+=1
  info=r.get('info')
  if isinstance(info,dict) and ('error' in info or 'exception' in info):errors+=1
  if 'Failed to establish' in line or 'Connection refused' in line:errors+=1
 dup=len(qids)-len(set(qids));
 if rs: rows.append({'file':p,'n':len(rs),'bad_json':bad,'empty_predictions':empty,'errors':errors,'duplicate_qids':dup,'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest()})
# classify known main artifacts
for r in rows:
 f=os.path.basename(r['file']); r['status']='candidate'
 if r['errors'] or (r['n'] and r['empty_predictions']==r['n']):r['status']='exclude_service_failure'
 if r['duplicate_qids'] or r['bad_json']:r['status']='quarantine_integrity'
 if f.startswith(('h3_e2e_','e2e_')) and r['status']=='candidate':r['status']='exploratory_valid'
 if f.startswith(('e4_','e13_','g4_')) and r['status']=='candidate':r['status']='legacy_or_main_candidate'
json.dump(rows,open(os.path.join(RES,'audit_results.json'),'w'),ensure_ascii=False,indent=2)
with open(os.path.join(RES,'audit_results.md'),'w') as out:
 out.write('# 结果反伪造审计\n\n审计规则：只依据逐题 JSONL、qid 唯一性、预测字段、错误字段和文件 hash；汇总 JSON 中的数字若不能回溯到逐题文件，不进入主表。\n\n|文件|n|空预测|错误|重复qid|状态|SHA256前12位|\n|---|---:|---:|---:|---:|---|---|\n')
 for r in rows:out.write(f"|{r['file']}|{r['n']}|{r['empty_predictions']}|{r['errors']}|{r['duplicate_qids']}|{r['status']}|{r['sha256'][:12]}|\n")
print('audited',len(rows))
