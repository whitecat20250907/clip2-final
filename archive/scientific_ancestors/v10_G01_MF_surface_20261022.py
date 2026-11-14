"""Extend the officially helpful surface representation to frozen full-TRAIN MF."""
from pathlib import Path
import os,sys,json,csv,time,shlex,traceback,subprocess,collections
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
sys.path.insert(0,str(Path(__file__).parent))
import v10_new_information_20261022 as G
import numpy as np
T=G.T;P=G.P;N=G.N;J=G.J;D=G.D/'G01_MF';D.mkdir(exist_ok=True);H=D/'readout';H.mkdir(exist_ok=True);FM=G.F.FM;REMOTE='/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_G01_MF_surface_20261022';SEEDS=G.SEEDS
def prepare():
 if (H/'inputs_report.json').exists():return
 old=T.read(T.V/'current_best_own_valid.json');assert old['submission_id']==587113,old['submission_id'];T.save(D/'own_reference.json',old);rules=T.read(G.D/'contraction_audit.json')['expansion'];align=G.alignment();pilot=T.V/'new_angles_20261021/music_witness_stage1';full=T.V/'new_angles_20261021/music_witness_full_train';records={}
 for folder,roles in [(pilot,list(T.read(pilot/'parts.json'))),(full,list(T.read(full/'parts.json')))]:
  for role in roles:
   for r in map(json.loads,(folder/'results'/role/'evidence.jsonl').read_text().splitlines()):
    key=r['split'],r['signal_id'],r['stream'];assert key not in records;records[key]=r
 t=P.F.arr(FM/'train.npz');v=P.F.arr(FM/'valid.npz');assert len(t['signal_ids'])==16047 and len(v['signal_ids'])==2117 and 'target' not in v;meta={r['signal_id']:r for r in T.read(pilot/'valid_metadata_features_only.json')};audit={}
 with (H/'token_provenance.jsonl').open('w',encoding='utf8') as out:
  for z,split,prompts in [(t,'train',t['prompts']),(v,'valid',[meta[str(s)]['prompt'] for s in v['signal_ids']])]:
   extra=np.zeros((len(z['signal_ids']),12,17),np.float32);counts=collections.Counter()
   for i,(sid,prompt) in enumerate(zip(z['signal_ids'].astype(str),prompts)):
    n=len(prompt.split());assert n==int(z['mask'][i].sum());blocks=[]
    for stream in ['A1L','A1R','A2L','A2R']:
     r=records[split,sid,stream];fields,prov=G.norm_fields(prompt,r['recognized_words'],rules,align);blocks.append(fields);counts[stream+'_changed_hits']+=abs(float(fields[:,0].sum())-align.align_words(prompt,' '.join(r['recognized_words']))['hits'])>1e-6;out.write(json.dumps(dict(split=split,signal_id=sid,stream=stream,**prov))+'\n')
    extra[i,:n]=np.concatenate(blocks+[np.asarray([len(rules.get(w,w).split())>1 for w in prompt.split()],np.float32)[:,None]],1)
   znew=dict(z,x=np.concatenate([z['x'],extra],2));assert np.array_equal(znew['x'][:,:,:55],z['x']);np.savez_compressed(H/(split+'.npz'),**znew);audit[split]=dict(counts,rows=len(z['signal_ids']))
 ids,base=G.F.csvpred(old['csv']);assert np.array_equal(ids,v['signal_ids']);mf=np.mean([P.F.arr(FM/'full'/f'witness_full_s{s}/valid.npz')['prediction'] for s in SEEDS],0);mos=np.mean([P.F.arr(G.H/'results'/f'G01_s{s}/valid.npz')['prediction'] for s in SEEDS],0);_,mfown=G.F.csvpred(T.read(G.MO/'own_reference.json')['csv']);replay=.9*mfown+.1*mos;drift=float(np.max(np.abs(base-replay)));assert drift<2e-7,drift;np.savez_compressed(H/'fixed_components.npz',signal_ids=ids,base=base,old_mf=mf)
 protocol=dict(UTC=T.utc(),task_id='G01-MF',reason='G01 MOSS surface fields officially improved587113; apply identical preregistered clear-contraction/provenance representation to previously audited MF, without changing recognizer or audio.',TRAIN=16047,VALID=2117,input_dimensions=72,original_dimensions=55,raw55_bitwise_preserved=True,word_count_and_human_targets_unchanged=True,tokenLP_duplicated=False,hidden=8,parameters=593,seeds=SEEDS,epochs=80,lr=.001,WD=.01,batch_max=128,fusion='Frozenown587113 +0.09*(normalizedMF -oldMF); keep P2 and normalizedMOSS exact',official_candidates=1,no_TRAIN_holdout_gate=True,EVAL_submission=False,baseline_replay_max_abs=drift);T.save(D/'protocol_lock.json',protocol);T.save(H/'inputs_report.json',dict(protocol,audit=audit,provenance_SHA=T.sha(H/'token_provenance.jsonl'),VALID_human_targets=False))
 with T.locked() as ok:
  assert ok;plan=T.read(G.B/'execution_protocol.json')
  if 'G01-MF' not in {r['id'] for r in plan['tasks']}:plan['tasks'].append(dict(id='G01-MF',priority='P0 高',task='将已胜出的明确缩写/来源修正扩到MF，保留当前归一化MOSS',state='缓存重建完成，四种子小头待接轻载卡',ETA='小头数分钟，核验及一份官测约15–35分钟；胜出再固定联合',dependency='16047清理TRAIN既有MF证据，不重新提取音频，原P2及G01 MOSS固定',source_id='用户继续全部任务；G01官方提升后的同规则MF扩展',source_plan=str(D/'protocol_lock.json')))
  T.save(G.B/'execution_protocol.json',plan)
 T.refresh()
def deploy():
 if (H/'registered.json').exists():return
 roles=[f'witness_full_s{s}' for s in SEEDS];p=dict(owner_uid=1030,roles=roles,expected_train_rows=16047,policy=T.read(D/'protocol_lock.json'));env=dict(MKL_THREADING_LAYER='SEQUENTIAL',TMPDIR=REMOTE+'/private_cache/tmp',CUDA_CACHE_PATH=REMOTE+'/private_cache/cuda',XDG_CACHE_HOME=REMOTE+'/private_cache');fleet=dict(root=P.L.R['4090'],owner_uid=1030,tasks=[dict(id=r,worker='worker.py',python=P.L.PY['4090'],args=['--base',REMOTE,'--role',r],free_MiB=3072,allow_owned_share=True,env=env) for r in roles]);ctrl=(J/'v10_revision_controller_20261020.py').read_text().replace("pending=list(p['tasks']);","(b/'tasks').mkdir(exist_ok=True)\nfor e in p['tasks']:(b/'tasks'/e['id']).mkdir(exist_ok=True)\npending=list(p['tasks']);");files={'protocol.json':json.dumps(p).encode(),'fleet.json':json.dumps(fleet).encode(),'worker.py':(J/'v10_music_witness_head_worker_20261021.py').read_bytes(),'controller.py':ctrl.encode(),'train.npz':(H/'train.npz').read_bytes(),'valid.npz':(H/'valid.npz').read_bytes()};reg=G.A.deploy_bundle('4090',REMOTE,files);T.save(H/'registered.json',dict(reg,roles=roles,UTC=T.utc()))
def collect():
 s=N.rpc('4090','from pathlib import Path\nimport json\nb=Path('+repr(REMOTE)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["progress"]={r:json.loads((b/"tasks"/r/"progress.json").read_bytes()) for r in s["active"] if (b/"tasks"/r/"progress.json").exists()};s["errors"]={r:(b/r/"worker.log").read_text(errors="replace")[-2200:] for r,q in s["done"].items() if q["exit_code"]};print(json.dumps(s))');T.save(H/'status.json',dict(s,UTC=T.utc()));assert not s['errors'],s['errors']
 for role in s['done']:
  d=H/'results'/role;d.mkdir(parents=True,exist_ok=True)
  if (d/'result.json').exists():continue
  for name in ['valid.npz','model.pt','result.json']:(d/name).write_bytes(N.run('4090','cat '+shlex.quote(REMOTE+'/tasks/'+role+'/'+name)))
 return len(s['done'])==4
def candidate():
 z=P.F.arr(H/'fixed_components.npz');v=P.F.arr(H/'valid.npz');mf=np.mean([P.F.arr(H/'results'/f'witness_full_s{s}/valid.npz')['prediction'] for s in SEEDS],0);pred=z['base']+.09*(mf-z['old_mf']);assert np.array_equal(z['signal_ids'],v['signal_ids']) and np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all();out=H/'candidate';out.mkdir(exist_ok=True)
 with (out/'valid_predictions.csv').open('w',newline='') as f:csv.writer(f).writerows(zip(z['signal_ids'],[format(x*100,'.10f') for x in pred]))
 report=dict(UTC=T.utc(),csv_SHA=T.sha(out/'valid_predictions.csv'),protocol=T.read(D/'protocol_lock.json'),inputs=T.read(H/'inputs_report.json'),VALID_human_targets=False,EVAL_submission=False);T.save(out/'report.json',report);T.save(out/'verification.json',dict(status='verified',csv_sha256=report['csv_SHA'],report_sha256=T.sha(out/'report.json')));cid='v10_G01_MF_surface_fixed9pct_20261022';c=P.H.F.candidate(out,cid,cid,'Same fullTRAIN MF16,047 and4seeds;append17clearcontraction/provenancefields,keepnative55;replaceMF9%only in own587113;officialadoption.');P.H.F.record=P.H.record;spec=P.H.F.submit_candidates([c],H/'official');T.save(H/'official_registered.json',dict(UTC=T.utc(),spec=spec))
def main():
 prepare();deploy();T.save(D/'watch_registered.json',dict(UTC=T.utc(),pid=os.getpid()));end=time.monotonic()+4*3600
 while time.monotonic()<end:
  try:
   if not (H/'official_registered.json').exists():
    done=collect();s=T.read(H/'status.json');G.F.progress('G01-MF',f"MF缩写修正四种子：活跃{len(s['active'])}，完成{len(s['done'])}/4，排队{len(s['pending'])}",'小头数分钟；固定9%MF替换候选核验和官测约15–35分钟，平台延迟另计')
    if not done:time.sleep(10);continue
    candidate()
   P.H.F.record=P.H.record
   if not (H/'complete.json').exists():
    if not P.H.F.poll_official(H/'official',T.read(H/'official_registered.json')['spec']):time.sleep(15);continue
    rr=T.read(H/'candidate/official_result.json');old=T.read(T.V/'current_best_own_valid.json');adopt=rr['result']['RMSE']<old['RMSE']-1e-12
    if adopt:T.save(T.V/'current_best_own_valid.json',dict(old,UTC=T.utc(),submission_id=rr['submission_id'],RMSE=rr['result']['RMSE'],Correlation=rr['result']['Correlation'],csv=rr['csv'],csv_SHA=rr['csv_SHA'],receipt=rr['receipt'],previous_best=old,MF_surface_stage=str(D),scope='MF9%surfacefields;keepP2andG01MOSS',full_raw_infer_all_complete=False,frozen_infer_package_not_updated=True))
    T.save(H/'complete.json',dict(UTC=T.utc(),official=rr,adopted=adopt,gain_RMSE=old['RMSE']-rr['result']['RMSE']))
   result=T.read(H/'complete.json');current=T.read(T.V/'current_best_own_valid.json')
   if result['adopted'] and current['submission_id']==result['official']['submission_id']:
    import v10_music_full_train_20261021 as R
    R.H=H
    if not R.joint('v10_G01_MF'):time.sleep(15);continue
   if 'G01-MF' in {r['id'] for r in T.read(T.STATE)['tasks']}:T.mark_complete('G01-MF',[H/'complete.json',D/'protocol_lock.json'])
   T.save(D/'pipeline_complete.json',dict(UTC=T.utc(),result=result));import v10_score_scope_publish_20261021 as S;S.main();return
  except (OSError,EOFError,TimeoutError,subprocess.TimeoutExpired) as ex:T.save(D/'retry.json',dict(UTC=T.utc(),error_type=type(ex).__name__));time.sleep(15)
 raise TimeoutError('MF surface pipeline exceeded4h')
if __name__=='__main__':
 try:main()
 except BaseException as ex:T.save(D/'needs_attention.json',dict(UTC=T.utc(),error=repr(ex),traceback=traceback.format_exc()));raise
