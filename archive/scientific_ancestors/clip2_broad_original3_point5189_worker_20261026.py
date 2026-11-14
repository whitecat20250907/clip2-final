"""Separate-stage MF/AF3/Phi point tape, using only original frozen scorers.

Scientific probability-coverage gates are waived by the new diagnostic stage.
The raw eleven events, complete-numeral/native-EOS semantics, original prompts,
audio, checkpoint and precision remain unchanged. Default is model-free audit.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,contextlib,hashlib,importlib.util,json,math,os,re,statistics,sys,time,traceback

STAGE='BROAD_POINT_20261026'
OWNER=Path('/home/douhan/zhaojiahao/b1_parallel/zhaojiahao_1')
OLD=OWNER/'runs/clip2_jury_20261024'
BASE=OWNER/'runs/clip2_broad_trials_20261026'
AUTH_SHA='c7f384dc70a6e2f464ccb8ac6ce049bd041cabc5d3304b69be606ec3967d824c'
HELPER_SHA='87fa3943fb40bf598c2b78f52af5a167a87a322e397f3d26863c02999340d722'
QC_SHA='5c81f3f8d7bc822f59a5af39ced153a9f6b7bc73b7cb226b84bc5dd42a95da30'
ROWS_SHA='450e31615ac2dec81f823680d34f9444ca197075ab3b13fdf293638459942ec6'
INPUT_SHA='3278451ae60f6403bb71b85a39e9655e5d8263028fa331bf25be8c1a9822782a'
CORE_SHA='d6acb30cbf6cf8b909e1bfb5fa07c9723ce36a5060c36aaf5dbf3988e4f4abe1'
SEATS={
 'MF':dict(kind='core',worker='jury_mf_moss_worker.py',worker_SHA=CORE_SHA,binding='MF_MOSS_QC64_runtime_manifest.json',binding_SHA='2e68dbe67a31eca17065fe71dc008f608c394071888b52d0d2d823a7406658db',python='/home/douhan/miniforge3/envs/clip2/bin/python'),
 'AF3HF':dict(kind='native',worker='jury_priority3_qc64_worker.py',worker_SHA='b76ac76f685ee194a8df2fb1702f7fbe21f319944a296dc9c0da784460973f22',binding='AF3HF_QC64_binding.json',binding_SHA='e8253748883985093eaeae93ac4c0f6d7a210c0f0b3461d31178a2185218a11c',adapter_SHA='206f1c8dbbf46f7ec50e43748fd78e1b3387f9000421f845183461bc423eba74',python=str(OLD/'envs/AF3HF_v2/bin/python')),
 'Phi4MM':dict(kind='native',worker='jury_optional_mi_phi_qc64_worker.py',worker_SHA='6136626970419682f20d86cc93dea913ad42c53b91c79a0acca94fd29a7225af',binding='Phi4MM_QC64_binding_zerooutput_v2.json',binding_SHA='612407956bc1082808208fa526c338db305fdbba63d5ab4e8f71e8852e5004f0',adapter_SHA='d2c1673e81fd7c415c3bdefeff10e4ce7f1132b0882b7077f1bab979b43f688f',python=str(OLD/'envs/Phi4MM_private/bin/python')),
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def owned(p):
 p=Path(p).resolve();assert p.is_relative_to(OWNER);return p
def checked(r):
 p=owned(r['path']);assert sha(p)==r['SHA'],str(p);return p
def load(name,p,h):
 assert sha(p)==h
 spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prepare(a):
 cf=owned(a.config);assert cf.is_relative_to(BASE)and sha(cf)==a.config_sha;c=read(cf)
 assert c['schema']=='CLIP2.broad.original3.point5189.stage.v1'and c['stage_ID']==STAGE
 assert c['owner_UID']==os.getuid()==1004 and owned(c['stage_base'])==BASE
 assert c['worker_SHA']==sha(__file__)and c['judge']in SEATS
 judge=c['judge'];seat=SEATS[judge]
 assert Path(sys.executable).resolve()==Path(seat['python']).resolve(),'Original private numerical environment required'
 helper=checked(c['shared_Q2_helpers']);assert sha(helper)==HELPER_SHA
 U=load('broad_original3_shared_nonmodel_helpers',helper,HELPER_SHA)
 auth=read(checked(c['authorization']));assert c['authorization']['SHA']==AUTH_SHA
 for flag in ['execution_authorized','scientific_QC_meta_CV_ladder_and_adoption_gates_waived_for_diagnostic_trials','old_closed_scientific_routes_may_be_reused_or_completed_only_as_new_registered_diagnostics','partial_original_stream_completion_permitted_in_new_stage','no_live_process_restart_or_foreign_process_signal','no_EVAL_submission']:
  assert auth[flag]is True,flag
 rev=read(checked(c['independent_review']));assert rev['accepted']is True and rev['approved_worker_SHA']==sha(__file__)
 assert c['scope']==dict(TRAIN_rows=3072,VALID_rows=2117,EVAL_rows=0,rows=5189,measurements=62268)
 assert c['seed']==20261024 and 0<c['cutoff_epoch']-c['T0_epoch']<=48*3600
 assert set(c['allowed_GPU_indices'])<={1,2,3,4,6,7}and c['allowed_GPU_indices']
 for ref in c['historical_closures']:checked(ref)
 qc_file=checked(c['original_QC_protocol']);assert qc_file==OLD/'QC64_protocol.json'and sha(qc_file)==QC_SHA
 binding_file=checked(c['original_binding']);assert binding_file==OLD/'runtime'/seat['binding']and sha(binding_file)==seat['binding_SHA']
 original_file=checked(c['original_QC_worker']);assert original_file==OLD/'runtime'/seat['worker']and sha(original_file)==seat['worker_SHA']
 W=load('broad_original3_frozen_'+judge+'_preparation',original_file,seat['worker_SHA'])
 args=SimpleNamespace(base=str(OLD),protocol=str(qc_file),protocol_sha=QC_SHA,host='3090',judge=judge,reduce_only=True,audit_only=True,gpu=None,shards=1,shard_index=0,binding=str(binding_file),binding_sha=seat['binding_SHA'],runtime_manifest=str(binding_file),runtime_manifest_sha=seat['binding_SHA'])
 if seat['kind']=='core':
  _,p,_=W.protocol_inputs(args);core=W;binding=read(binding_file);model=p['models'][judge];sources=p['source_SHA']
 else:
  _,p,binding,core,_=W.protocol_and_tasks(args);model=binding['model'];sources=binding['source_SHA']
  assert sources['adapter']==seat['adapter_SHA'];checked(dict(path=str(OLD/binding['source_files']['adapter']),SHA=seat['adapter_SHA']))
 assert sha(OLD/'runtime/jury_mf_moss_worker.py')==CORE_SHA
 for ref in c.get('additional_model_file_refs',[]):checked(ref)
 row_file=checked(c['original_point_rows_protocol']);assert row_file==OLD/'Qwen2Audio_unionEOS_POINT7378_protocol.json'and sha(row_file)==ROWS_SHA
 point=read(row_file);rows=point['rows'][:5189]
 assert point['prompts']==read(qc_file)['prompts']and point['prompts_SHA']==read(qc_file)['prompts_SHA']
 inp=point['input_manifest'];input_file=owned(OLD/inp['path']);assert sha(input_file)==inp['SHA']==INPUT_SHA
 inputs=read(input_file)['rows'][:5189]
 assert len(rows)==len(inputs)==5189 and len({r['signal_id']for r in rows})==5189
 assert [sum(r['split']==s for r in rows)for s in ['train','valid']]==[3072,2117]
 assert all(r['split']=='train'for r in rows[:3072])and all(r['split']=='valid'for r in rows[3072:])
 tasks=[]
 for index,(row,source)in enumerate(zip(rows,inputs)):
  path=Path(row['audio_path']).resolve();assert path==Path(p['data_roots'][row['split']]).resolve()/'signals'/(row['signal_id']+'.flac')
  assert (row['signal_id'],row['split'],row['audio_path'])==(source['signal_id'],source['split'],source['audio_path'])
  st=path.stat();assert st.st_size==source['bytes']and st.st_mtime_ns==source['mtime_ns']
  for prompt in p['prompts']:
   for ear in [0,1]:tasks.append(dict(row=row,row_index=index,prompt=prompt,ear=ear,control='real',audio_SHA=source['audio_SHA']))
 assert len(tasks)==62268 and len(p['prompts'])==6
 jobs=c['batches'];seen=[]
 for job in jobs:
  assert re.fullmatch(r'[A-Za-z0-9_-]+',job['job_ID'])and 0<=job['task_start']<job['task_end']<=62268
  assert job['task_end']-job['task_start']<=3072
  seen.extend(range(job['task_start'],job['task_end']))
 assert seen==list(range(62268))and len({j['job_ID']for j in jobs})==len(jobs)
 if judge=='Phi4MM':
  assert 8<=jobs[0]['task_end']<=32 and jobs[0]['production_timing_only']is True
 identity=hashlib.sha256(U.canonical(dict(stage_ID=STAGE,judge=judge,QC_protocol_SHA=QC_SHA,binding_SHA=seat['binding_SHA'],original_numeric_sources=sources,model=model,point_rows_SHA=ROWS_SHA,input_manifest_SHA=INPUT_SHA,scope='TRAIN3072_VALID2117',rawmass_scientific_gate='waived_positive_finite_native_probability_only',seed=20261024))).hexdigest()
 return c,U,core,p,binding,model,sources,tasks,identity

def expectation(U,m,model):
 value=U.expectation(m)
 assert m['numeral_token_ids']==model['expected_numeral_token_ids']
 expected_eos=model['termination_policy']['stop_ids']if 'termination_policy'in model else model['native_EOS']
 assert m['native_EOS']==expected_eos
 assert m['uncached_full_forward']is True and m['context_forward_count']==1
 return value

class Tape:
 def __init__(self,a,z):
  self.a=a;self.c,self.U,self.core,self.p,self.binding,self.model,self.sources,self.tasks,self.identity=z
  self.judge=self.c['judge'];self.old=self.core.Store(OLD,QC_SHA,self.judge)
 def key(self,t):return hashlib.sha256(self.U.canonical(dict(stage_ID=STAGE,identity_SHA=self.identity,judge=self.judge,signal_id=t['row']['signal_id'],prompt_id=t['prompt']['id'],ear=t['ear']))).hexdigest()
 def path(self,t):return BASE/'measurements'/self.judge/(self.key(t)+'.json')
 def existing(self,t):
  path=self.path(t)
  if not path.exists():return None
  q=read(path)
  assert q['complete']is True and q['stage_config_SHA']==self.a.config_sha and q['identity_SHA']==self.identity and q['measurement_key']==self.key(t)
  assert q['judge']==self.judge and q['signal_id']==t['row']['signal_id']and q['split']==t['row']['split']and q['prompt_id']==t['prompt']['id']and q['ear']==t['ear']
  assert q['audio_source_SHA']==t['audio_SHA']and q['prompt_SHA']==hashlib.sha256(t['prompt']['text'].encode()).hexdigest()
  assert q['diagnostic']==expectation(self.U,q['measurement'],self.model)
  return q,dict(path=str(path),SHA=sha(path),origin='new_diagnostic_stage')
 def source(self,t):
  cached=self.existing(t)
  if cached:return cached
  old=self.old.existing(t)
  if old is None:return None
  assert old['phase']=='qc64'and old['source_SHA']==self.sources and old['model_spec_SHA']==hashlib.sha256(self.U.canonical(self.model)).hexdigest()
  m=old['measurement'];assert m['direction']==t['prompt']['direction']
  path=self.old.record_path(t)
  return dict(measurement=m,diagnostic=expectation(self.U,m,self.model)),dict(path=str(path),SHA=sha(path),origin='original_QC_real_raw_reused_without_remeasurement')

def export(a,z):
 c,U,core,p,binding,model,sources,tasks,identity=z;tape=Tape(a,z);judge=c['judge'];d=BASE/'exports'/judge
 with U.lock(BASE/'leases'/('export_'+judge+'.lock')):
  done=d/'complete.json'
  if done.exists():
   q=read(done);assert q['stage_config_SHA']==a.config_sha
   for ref in q['artifacts']:checked(ref)
   print(json.dumps(dict(reused_complete=True,complete=str(done))));return
  assert not(d/'export_intent.json').exists(),'Unknown export intent: preserve and inspect, no retry'
  U.publish(d/'export_intent.json',dict(UTC=U.utc(),PID=os.getpid(),stage_config_SHA=a.config_sha,worker_SHA=sha(__file__),no_GPU_fit_CV_API=True))
  import numpy as np
  scores=np.full((5189,6,2),np.nan);masses=np.full_like(scores,np.nan);index=[];reused=0
  promptids=[x['id']for x in p['prompts']];pi={x:i for i,x in enumerate(promptids)}
  for t in tasks:
   src=tape.source(t);assert src is not None,'Incomplete original diagnostic tape; never impute'
   q,r=src;i=t['row_index'];j=pi[t['prompt']['id']];e=t['ear'];scores[i,j,e]=q['diagnostic']['oriented_score'];masses[i,j,e]=q['diagnostic']['raw_total_mass']
   index.append(dict(row_index=i,signal_id=t['row']['signal_id'],prompt_id=t['prompt']['id'],ear=e,record=r))
   reused+=r['origin']!='new_diagnostic_stage'
  assert np.isfinite(scores).all()and np.isfinite(masses).all()and(masses>0).all()
  rows=[t['row']for t in tasks[::12]];raw=d/'raw_fields.npz';tmp=d/'raw_fields.tmp.npz';assert not raw.exists()and not tmp.exists()
  with tmp.open('xb')as f:
   np.savez_compressed(f,signal_ids=np.array([r['signal_id']for r in rows]),split=np.array([r['split']for r in rows]),severity=np.array([r['severity']for r in rows]),prompt_ids=np.array(promptids),score_ears=scores,raw_prompt_fields=scores.mean(2),raw6mean=scores.mean((1,2)),raw_mass_ears=masses);f.flush();os.fsync(f.fileno())
  os.link(tmp,raw);tmp.unlink();ix=d/'measurement_index.json';U.publish(ix,dict(stage_ID=STAGE,identity_SHA=identity,rows=5189,records=index))
  artifacts=[dict(path=str(f),SHA=sha(f))for f in [raw,ix]]
  U.publish(done,dict(UTC=U.utc(),schema='CLIP2.broad.point5189.raw_complete.v1',stage_ID=STAGE,judge=judge,complete=True,rows=5189,TRAIN_rows=3072,VALID_rows=2117,EVAL_rows=0,measurements=62268,identity_SHA=identity,stage_config_SHA=a.config_sha,worker_SHA=sha(__file__),raw_fields_SHA=sha(raw),measurement_index_SHA=sha(ix),original_reused_records=reused,new_diagnostic_records=62268-reused,technical_complete=True,scientific_control_gates_waived=True,no_original_gate_recomputation=True,no_model_fit_CV_API_or_EVAL=True,artifacts=artifacts))

def execute(a,z):
 c,U,core,p,binding,model,sources,tasks,identity=z;judge=c['judge'];tape=Tape(a,z)
 job=next(j for j in c['batches']if j['job_ID']==a.job);assigned=tasks[job['task_start']:job['task_end']];d=BASE/'jobs'/a.job
 assert a.gpu in c['allowed_GPU_indices']and time.time()<c['cutoff_epoch']
 with contextlib.ExitStack()as stack:
  stack.enter_context(U.lock(BASE/'leases/jobs'/(a.job+'.lock')))
  assert not(d/'registration.json').exists(),'Existing registered batch: recover original writer; never restart'
  initial=U.gpu_state(a.gpu);lease=BASE/'leases'/('GPU_'+initial['uuid']+'.lock')
  stack.enter_context(U.lock(lease,a.gpu_lease_fd))
  first=U.gpu_state(a.gpu);time.sleep(1);second=U.gpu_state(a.gpu)
  assert first['uuid']==second['uuid']==initial['uuid']and not first['computePIDs']and not second['computePIDs']
  assert min(first['memory_free_MiB'],second['memory_free_MiB'])>=22600 and max(first['utilization'],second['utilization'])<=25
  os.environ.update(CUDA_VISIBLE_DEVICES=second['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  for name in ['HF_HOME','HF_MODULES_CACHE','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR']:
   folder=d/'cache'/name;folder.mkdir(parents=True,exist_ok=True);os.environ[name]=str(folder)
  U.publish(d/'registration.json',dict(UTC=U.utc(),PID=os.getpid(),UID=os.getuid(),argv=sys.argv,stage_ID=STAGE,judge=judge,job=job,GPU=second,first_GPU=first,stage_config_SHA=a.config_sha,worker_SHA=sha(__file__),identity_SHA=identity,no_training_QC_CV_API_or_schedule=True))
  session=None;start=time.monotonic();newtimes=[];finished=0;reused=0
  for t in assigned:
   if tape.source(t)is not None:finished+=1;reused+=1;continue
   if time.time()>=c['cutoff_epoch']:break
   with U.lock(BASE/'leases/measurement_keys'/judge/(tape.key(t)+'.lock')):
    if tape.source(t)is not None:finished+=1;reused+=1;continue
    if session is None:
     if judge=='MF':session=core.NumericSession(OLD,p,judge,d)
     else:
      adapter=load('broad_original3_frozen_native_'+judge,OLD/binding['source_files']['adapter'],SEATS[judge]['adapter_SHA'])
      session=adapter.NativeNumericSession(judge,model,20261024,.8,lambda name,q:U.publish(d/name,dict(UTC=U.utc(),**q)))
    if time.time()>=c['cutoff_epoch']:break
    now=U.gpu_state(a.gpu);assert now['uuid']==second['uuid']and set(now['computePIDs'])<={os.getpid()},'Unexpected compute PID: preserve partials, no signal/retry'
    before=time.monotonic();measurement=session.score(t);elapsed=time.monotonic()-before
    assert measurement['direction']==t['prompt']['direction'];diagnostic=expectation(U,measurement,model)
    q=dict(UTC=U.utc(),complete=True,stage_ID=STAGE,judge=judge,stage_config_SHA=a.config_sha,identity_SHA=identity,measurement_key=tape.key(t),signal_id=t['row']['signal_id'],split=t['row']['split'],row_index=t['row_index'],prompt_id=t['prompt']['id'],prompt_SHA=hashlib.sha256(t['prompt']['text'].encode()).hexdigest(),ear=t['ear'],control='real',audio_source_SHA=t['audio_SHA'],original_binding_SHA=SEATS[judge]['binding_SHA'],source_SHA=sources,model_spec_SHA=hashlib.sha256(U.canonical(model)).hexdigest(),worker_SHA=sha(__file__),measurement=measurement,diagnostic=diagnostic,wall_measurement_seconds=elapsed)
    U.publish(tape.path(t),q);newtimes.append(elapsed);finished+=1
    steady=newtimes[2:]if len(newtimes)>2 else newtimes
    U.publish(d/'progress.json',dict(UTC=U.utc(),PID=os.getpid(),judge=judge,job_ID=a.job,assigned_measurements=len(assigned),completed=finished,new_measurements=len(newtimes),original_or_stage_reused=reused,elapsed_seconds=time.monotonic()-start,warm_measurement_seconds=newtimes[:2],steady_sample_count=len(steady),steady_median_seconds=statistics.median(steady),remaining_job_eta_seconds=(len(assigned)-finished)*statistics.median(steady),scope_serial_projection_seconds=(62268-finished)*statistics.median(steady),ETA_is_measured_projection_not_completion_promise=True),True)
  remaining=sum(tape.source(t)is None for t in assigned)
  steady=newtimes[2:]if len(newtimes)>2 else newtimes
  U.publish(d/'terminal.json',dict(UTC=U.utc(),PID=os.getpid(),stage_ID=STAGE,judge=judge,job_ID=a.job,stage_config_SHA=a.config_sha,worker_SHA=sha(__file__),batch_complete=remaining==0,assigned_measurements=len(assigned),remaining_measurements=remaining,new_measurements=len(newtimes),reused_measurements=reused,production_timing_only=job.get('production_timing_only',False),warm_measurement_seconds=newtimes[:2],steady_sample_count=len(steady),steady_median_seconds=statistics.median(steady)if steady else None,scope_serial_projection_seconds=(62268-finished)*statistics.median(steady)if steady else None,no_QC_CV_model_fit_API_or_EVAL=True))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True);ap.add_argument('--job');ap.add_argument('--gpu',type=int);ap.add_argument('--gpu-lease-fd',type=int);ap.add_argument('--execute',action='store_true');ap.add_argument('--export-only',action='store_true');a=ap.parse_args();z=prepare(a)
 if a.export_only:export(a,z);return
 if not a.execute:print(json.dumps(dict(audit_passed=True,judge=z[0]['judge'],scope=z[0]['scope'],registered_batches=len(z[0]['batches']),no_model_GPU_measurement_fit_CV_API_or_writes=True)));return
 assert a.job and a.gpu is not None
 try:execute(a,z)
 except BaseException as error:
  U=z[1];path=BASE/'jobs'/a.job/'attention.json'
  if not path.exists():U.publish(path,dict(UTC=U.utc(),PID=os.getpid(),kind='engineering_attention',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),partial_preserved=True,no_automatic_retry=True))
  raise

if __name__=='__main__':main()
