"""G01/G02 official-first execution and four new-information task registry."""
from pathlib import Path
import os,sys,json,time,csv,gzip,hashlib,shlex,subprocess,traceback,concurrent.futures,importlib.util,collections
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
sys.path.insert(0,str(Path(__file__).parent))
import v10_multilistener_fusion_20261021 as F
import numpy as np
T=F.T;P=F.P;N=F.N;A=F.A;J=F.J;B=F.B;D=T.V/'next_optimization_20261021';H=D/'G01_G02';H.mkdir(exist_ok=True);MO=F.MO;SEEDS=[142,242,342,442]

def registry():
 tasks=[dict(id='G01',priority='P0 最高',task='MOSS明确缩写的匹配字段与来源标记，固定10%替换官测',state='按固定28规则重建附加字段，原始55维与置信度保留',ETA='复用缓存，目标30–90分钟内首份官测，平台另计',dependency='3072训练身份、8维隐层、80epoch、四种子固定；只补明确缩写，不改目标或词数，不复制logprob'),dict(id='G02',priority='P0 最高',task='MOSS按最终固定融合目标训练，原输入与10%比例不变',state='匹配留出父预测及留出MOSS锚定，和G01独立并行',ETA='目标30–90分钟内官测，平台另计',dependency='损失MSE(.9b+.1m,y)+.01MSE(m,oldMOSS_OOF)；不反推无界目标；3072/80epoch/四种子固定'),dict(id='G03',priority='P1 高',task='MOSS对参考词及实际竞争词的同条件支持差评分',state='固定候选来源与共同前缀、边界；立即做32条来源接口',ETA='先32条实测约30–90分钟；首份官测预算8–16小时，按实测速率修订',dependency='MF/四Whisper实际错词提供最多1竞争词；同音频/前缀/边界，记录sum/mean/token长度与缺失；不跨模型减概率'),dict(id='G04',priority='P2 接续',task='MOSS末层音频表示固定汇总，补充原词证据',state='接口准备；优先使用前三项之外的空卡、轻载卡',ETA='首份预算6–12小时，先32条实测',dependency='固定编码器末层、均值/标准差/四时间块、固定小投影；不返回全部LLM层，不扩交叉注意力')]
 with T.locked() as ok:
  assert ok;plan=T.read(B/'execution_protocol.json');known={x['id'] for x in plan['tasks']}
  for task in tasks:
   if task['id'] not in known:plan['tasks'].append(dict(task,source_id='用户批准下一轮优化：优先补遗漏信息',source_plan=str(D/'下一轮优化空间与计划.md'),source_SHA=T.sha(D/'下一轮优化空间与计划.md')))
  for task in plan['tasks']:
   if task['id']=='F04':task.update(state='有条件停止：F03三份官测均未胜，不开展同配方内部替换',ETA='前置未满足，无当前运行日期')
   if task['id']=='OM04':task.update(priority='P3 候补',state='GPU来源资格及全量扩展暂不启动；权重下载/私有环境缓存保留，G01/G02/G03优先',ETA='当前不承诺官测日期，等待更具体的信息缺口')
  T.save(B/'execution_protocol.json',plan)
 z=T.read(B/'task_progress.json');z['tasks']['F04']=dict(state='F03三份官方均未胜；不开展内部替换，保留为条件停止，不算完成',ETA='无当前运行日期');z['tasks']['OM04']=dict(priority='P3 候补',state='按用户新优先级，AF-Next GPU资格及扩展已停住；下载/私有环境缓存保留',ETA='候补，不排当前官测日期');T.save(B/'task_progress.json',z);T.refresh()

def alignment():
 af=T.V.parent.parent/'research/personal_clip2/alignment.py';assert T.sha(af)==T.read(F.FM/'inputs_report.json')['machine_alignment_code_SHA'];s=importlib.util.spec_from_file_location('g01_fixed_align',af);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def expand(words,rules):
 tokens=[];orig=[];amb=[]
 for i,w in enumerate(words):
  w=w.casefold().replace('’',"'");pieces=rules.get(w,w).split();ambiguous=w.endswith(("'s","'d")) or w=="ain't"
  for token in pieces:tokens.append(token);orig.append(i);amb.append(ambiguous)
 return tokens,orig,amb

def norm_fields(prompt,hyp,rules,align):
 ref=prompt.casefold().split();rt,ri,ra=expand(ref,rules);ht,hi,ha=expand(hyp,rules);a=align.align_words(' '.join(rt),' '.join(ht));out=np.zeros((len(ref),4),np.float32);groups=collections.defaultdict(list)
 for j,r in enumerate(ri):groups[r].append(j)
 used=collections.Counter(hi[k] for k in a['hyp_indices'] if k is not None)
 for original,js in groups.items():
  out[original,0]=np.mean([a['labels'][j] for j in js]);out[original,1]=np.mean([a['hyp_indices'][j] is None for j in js]);ks=[a['hyp_indices'][j] for j in js if a['hyp_indices'][j] is not None];out[original,2]=any(used[hi[k]]>1 for k in ks);out[original,3]=any(ra[j] or not a['mapping_mask'][j] for j in js) or any(ha[k] for k in ks)
 return out,dict(normalized_reference=rt,reference_origin_indices=ri,normalized_hypothesis=ht,hypothesis_origin_indices=hi,ambiguous_hypothesis=ha,normalized_hyp_indices=a['hyp_indices'],reference_count_preserved=len(ref),native_logprob_copied=False)

def prepare():
 if (H/'inputs_report.json').exists():return
 rules=T.read(D/'contraction_audit.json')['expansion'];assert len(rules)==28;align=alignment();case,provenance=norm_fields('do not you tell me what to do',["don't",'you','tell','me','what','to','do'],rules,align);assert case[:,0].sum()==8 and case[:2,2].tolist()==[1.,1.];amb,_=norm_fields('he is here',["he's",'here'],rules,align);assert amb[:,3].max()==1;records={}
 for path in (B/'OM02_pilot_evidence/results').glob('part*/evidence.jsonl'):
  for line in path.read_text().splitlines():
   r=json.loads(line);key=(r['split'],r['signal_id'],r['stream']);assert key not in records;records[key]=r
 t=P.F.arr(MO/'train.npz');v=P.F.arr(MO/'valid.npz');v={k:v[k][:2117] for k in ['signal_ids','x','mask','parent']};validmeta={r['signal_id']:r for r in T.read(T.V/'new_angles_20261021/music_witness_stage1/valid_metadata_features_only.json')};parent=P.F.arr(MO/'parent2.npz');old=P.F.arr(MO/'paired_predictions.npz');assert np.array_equal(t['signal_ids'],parent['signal_ids']) and np.array_equal(t['signal_ids'],old['signal_ids']);t['parent']=parent['parent'].mean(0);t['anchor']=old['witness'].mean(0);v['anchor']=np.zeros(2117,np.float32);assert 'target' not in v;T.save(H/'parent_reference.json',T.read(MO/'own_reference.json'))
 audit={};provenance_file=H/'G01_token_provenance.jsonl'
 with provenance_file.open('w',encoding='utf8') as out:
  for z,split,prompts in [(t,'train',t['prompts']),(v,'valid',[validmeta[str(s)]['prompt'] for s in v['signal_ids']])]:
   extra=np.zeros((len(z['signal_ids']),12,17),np.float32);counts=collections.Counter()
   for i,(sid,prompt) in enumerate(zip(z['signal_ids'].astype(str),prompts)):
    n=len(prompt.split());assert n==int(z['mask'][i].sum());parts=[]
    for stream in ['A1L','A1R','A2L','A2R']:
     r=records[split,sid,stream];q,prov=norm_fields(prompt,r['recognized_words'],rules,align);parts.append(q);oldhits=align.align_words(prompt,' '.join(r['recognized_words']))['hits'];changed=abs(float(q[:,0].sum())-oldhits)>1e-6;counts[stream+'_rows']+=1;counts[stream+'_changed_hits']+=changed;out.write(json.dumps(dict(signal_id=sid,split=split,stream=stream,raw_recognized_words=r['recognized_words'],**prov))+'\n')
    reference_expansion=np.array([len(rules.get(w,w).split())>1 for w in prompt.split()],np.float32)[:,None];extra[i,:n]=np.concatenate(parts+[reference_expansion],1)
   g1=dict(z,x=np.concatenate([z['x'],extra],2));assert np.array_equal(g1['x'][:,:,:55],z['x']) and np.array_equal(g1['mask'],z['mask']);g2=dict(z);np.savez_compressed(H/('G01_'+split+'.npz'),**g1);np.savez_compressed(H/('G02_'+split+'.npz'),**g2);audit[split]=dict(counts)
 # Same fixed text audit on MF A1L; no human scores used for the audit.
 refs=dict(zip(t['signal_ids'].astype(str),t['prompts'].astype(str)));seen=set();mfcounts=collections.Counter()
 for path in [T.V/'new_angles_20261021/music_witness_stage1/results'/role/'evidence.jsonl' for role in T.read(T.V/'new_angles_20261021/music_witness_stage1/parts.json')]:
  for line in path.read_text().splitlines():
   r=json.loads(line);sid=r['signal_id']
   if r['split']!='train' or r['stream']!='A1L' or sid not in refs:continue
   assert sid not in seen;seen.add(sid);q,_=norm_fields(refs[sid],r['recognized_words'],rules,align);mfcounts['rows']+=1;mfcounts['changed_hits']+=abs(float(q[:,0].sum())-align.align_words(refs[sid],' '.join(r['recognized_words']))['hits'])>1e-6
 assert len(seen)==3072;T.save(H/'surface_audit.json',dict(UTC=T.utc(),MOSS=audit,MF_A1L=dict(mfcounts),human_scores_used=False,rules=rules,rules_SHA=hashlib.sha256(json.dumps(rules,sort_keys=True).encode()).hexdigest(),native_logprob_fields_unchanged=True,provenance_SHA=T.sha(provenance_file)))
 T.save(H/'protocol_lock.json',dict(UTC=T.utc(),G01=dict(input_dimensions=72,new_fields='Perstream norm_hit_fraction,norm_unmapped_fraction,shared_native_source,ambiguity;reference_expansion flag;17new. Original55 unchanged; no new tokenLP values.',hidden=8,parameters=593),G02=dict(input_dimensions=55,hidden=8,parameters=449,objective='MSE(.9*parentOOF+.1*MOSS,y)+.01*MSE(MOSS,oldMOSS_OOF)',parent_and_anchor_leaveout=True),TRAIN=3072,VALID=2117,epochs=80,lr=.001,weight_decay=.01,seeds=SEEDS,fusion='Each independently90%frozenMFown587054+10%replacementMOSS; no recursive10%',official_only_adoption=True,no_TRAIN_score_gate=True,no_EVAL_submission=True))
 T.save(H/'inputs_report.json',dict(UTC=T.utc(),TRAIN=3072,VALID=2117,raw55_preserved=True,reference_word_count_preserved=True,parent_OOF_source=str(MO/'parent2.npz'),anchor_OOF_source=str(MO/'paired_predictions.npz'),G01_provenance=str(provenance_file),model_scoring_targets_unchanged=True,files={p.name:T.sha(p) for p in H.glob('G0[12]_*.npz')}))

def deploy(host):
 path=H/(host+'_registered.json')
 if path.exists():return
 root=P.L.R[host];uid=1004 if host=='3090' else 1030;remote=root+'/runs/v10_G01_G02_20261022' if host=='3090' else '/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_G01_G02_20261022';roles=[f'{g}_s{s}' for g in ['G01','G02'] for s in SEEDS];roles=roles[::2] if host=='3090' else roles[1::2];proto=dict(owner_uid=uid,roles=roles,policy=T.read(H/'protocol_lock.json'));env=dict(MKL_THREADING_LAYER='SEQUENTIAL',TMPDIR=remote+'/private_cache/tmp',CUDA_CACHE_PATH=remote+'/private_cache/cuda',XDG_CACHE_HOME=remote+'/private_cache');fleet=dict(root=root,owner_uid=uid,tasks=[dict(id=r,worker='worker.py',python=P.L.PY[host],args=['--base',remote,'--role',r],free_MiB=3072,allow_owned_share=True,env=env) for r in roles]);controller=(J/'v10_revision_controller_20261020.py').read_text().replace("pending=list(p['tasks']);","(b/'tasks').mkdir(exist_ok=True)\nfor e in p['tasks']:(b/'tasks'/e['id']).mkdir(exist_ok=True)\npending=list(p['tasks']);");files={'protocol.json':json.dumps(proto).encode(),'fleet.json':json.dumps(fleet).encode(),'worker.py':(J/'v10_G01_G02_worker_20261022.py').read_bytes(),'controller.py':controller.encode()};files.update({p.name:p.read_bytes() for p in H.glob('G0[12]_*.npz')});reg=A.deploy_bundle(host,remote,files);T.save(path,dict(reg,roles=roles,UTC=T.utc()))

def collect(host):
 reg=T.read(H/(host+'_registered.json'));remote=reg['remote'];code='from pathlib import Path\nimport json\nb=Path('+repr(remote)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["errors"]={r:(b/r/"worker.log").read_text(errors="replace")[-2000:] for r,q in s.get("done",{}).items() if q["exit_code"]};print(json.dumps(s))';s=N.rpc(host,code);T.save(H/(host+'_status.json'),dict(s,UTC=T.utc()));assert not s['errors'],s['errors']
 for role in s['done']:
  out=H/'results'/role;out.mkdir(parents=True,exist_ok=True)
  if (out/'result.json').exists():continue
  for name in ['valid.npz','model.pt','result.json']:(out/name).write_bytes(N.run(host,'cat '+shlex.quote(remote+'/tasks/'+role+'/'+name)))
 return len(s['done'])==len(reg['roles'])

def candidate(arm):
 z=P.F.arr(H/(arm+'_valid.npz'));pred=np.mean([P.F.arr(H/'results'/f'{arm}_s{s}/valid.npz')['prediction'] for s in SEEDS],0);pred=.9*z['parent']+.1*pred;assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all();out=H/arm/'candidate';out.mkdir(parents=True,exist_ok=True)
 with (out/'valid_predictions.csv').open('w',newline='') as f:csv.writer(f).writerows(zip(z['signal_ids'],[format(x*100,'.10f') for x in pred]))
 report=dict(UTC=T.utc(),arm=arm,csv_SHA=T.sha(out/'valid_predictions.csv'),protocol=T.read(H/'protocol_lock.json'),inputs=T.read(H/'inputs_report.json'),VALID_human_targets=False,EVAL_submission=False);T.save(out/'report.json',report);T.save(out/'verification.json',dict(status='verified',csv_sha256=report['csv_SHA'],report_sha256=T.sha(out/'report.json')));cid='v10_'+arm+'_MOSS_fixed10pct_20261022';c=P.H.F.candidate(out,cid,cid,'FrozenMOSS native3072TRAIN,4seeds,replace10%expert;'+('append17clearcontraction/provenancefields,original55unchanged' if arm=='G01' else 'same55/same449params,finalfusionMSE+.01OOFanchor,nointercept')+';officialonlyadoption.');P.H.F.record=P.H.record;spec=P.H.F.submit_candidates([c],H/arm/'official');T.save(H/arm/'official_registered.json',dict(UTC=T.utc(),spec=spec))

def finish(arm):
 rr=T.read(H/arm/'candidate/official_result.json');old=T.read(T.V/'current_best_own_valid.json');adopt=rr['result']['RMSE']<old['RMSE']-1e-12
 if adopt:T.save(T.V/'current_best_own_valid.json',dict(old,UTC=T.utc(),submission_id=rr['submission_id'],RMSE=rr['result']['RMSE'],Correlation=rr['result']['Correlation'],csv=rr['csv'],csv_SHA=rr['csv_SHA'],receipt=rr['receipt'],previous_best=old,new_information_stage=str(D),new_information_arm=arm,full_raw_infer_all_complete=False,frozen_infer_package_not_updated=True))
 T.save(H/arm/'complete.json',dict(UTC=T.utc(),official=rr,adopted=adopt,gain_RMSE=old['RMSE']-rr['result']['RMSE']))

def main():
 registry();prepare();T.save(H/'watch_registered.json',dict(UTC=T.utc(),pid=os.getpid(),script=str(Path(__file__).resolve())));end=time.monotonic()+10*3600
 while time.monotonic()<end:
  try:
   for host in ['3090','4090']:deploy(host)
   if not all((H/g/'official_registered.json').exists() for g in ['G01','G02']):
    with concurrent.futures.ThreadPoolExecutor(2) as ex:done=list(ex.map(collect,['3090','4090']))
    for g in ['G01','G02']:F.progress(g,'两机四种子小头已接入空卡/轻载卡；输入与来源核验已通过','小头通常数分钟；随后立即提交固定候选，平台约5–20分钟另计')
    if not all(done):time.sleep(15);continue
    for g in ['G01','G02']:
     if not (H/g/'official_registered.json').exists():candidate(g)
   waiting=False;P.H.F.record=P.H.record
   for g in ['G01','G02']:
    if not (H/g/'complete.json').exists():
     if not P.H.F.poll_official(H/g/'official',T.read(H/g/'official_registered.json')['spec']):waiting=True;continue
     finish(g)
   if waiting:time.sleep(20);continue
   best=min(['G01','G02'],key=lambda g:T.read(H/g/'complete.json')['official']['result']['RMSE']);rr=T.read(H/best/'complete.json');current=T.read(T.V/'current_best_own_valid.json')
   if rr['adopted'] and rr['official']['submission_id']==current['submission_id']:
    import v10_music_full_train_20261021 as R
    R.H=H/best
    if not R.joint('v10_'+best):time.sleep(20);continue
   for g in ['G01','G02']:
    if g in {r['id'] for r in T.read(T.STATE)['tasks']}:T.mark_complete(g,[H/g/'complete.json',H/'protocol_lock.json'])
   T.save(H/'pipeline_complete.json',dict(UTC=T.utc(),results={g:T.read(H/g/'complete.json') for g in ['G01','G02']}));import v10_score_scope_publish_20261021 as S;S.main();return
  except (OSError,EOFError,TimeoutError,subprocess.TimeoutExpired) as ex:T.save(H/'retry.json',dict(UTC=T.utc(),error_type=type(ex).__name__));time.sleep(20)
 raise TimeoutError('G01/G02 10h exceeded; artifacts preserved')
if __name__=='__main__':
 try:main()
 except BaseException as ex:T.save(H/'needs_attention.json',dict(UTC=T.utc(),error=repr(ex),traceback=traceback.format_exc()));raise
