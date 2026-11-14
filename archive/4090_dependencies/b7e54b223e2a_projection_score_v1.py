import os,sys,pathlib,json,hashlib,time,ast,math,re
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'PROJ_TDT_PLAN_V1.json').read_text())
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import numpy as np
from functools import lru_cache
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def save(n,**v):
 with(R/n).open('xb')as f:np.savez_compressed(f,**v)
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
def defs(f,names):
 nodes=[n for n in ast.parse(pathlib.Path(f).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f),'exec'),globals())
start=time.monotonic()
for n,h in read(R/'PROJ_TDT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
assert read(R/'PROJ_TDT_ASR_TRAIN_TERMINAL_V1.json')['actual_exit_code']==0;seal=read(R/'PROJ_TDT_ADAPTERS_SEALED_V1.json');assert seal['additional_global_optimizer_steps']==2520 
hp=read(p['source_full_plan']);bp=read(hp['native_full_plan']);B=pathlib.Path(hp['native_full_root']);W=pathlib.Path(hp['word_full_root']);S=pathlib.Path(p['native_root'])/'readout_screen_v1'
with np.load(bp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
with np.load(p['micro_roles'],allow_pickle=False)as z:fit=z['FIT'].tolist();held=z['held'].tolist()
ss=fit+held;N=len(fit);assert N==4096 and len(held)==1024 and not set(ss)&set(dev);ix={s:i for i,s in enumerate(train)};ii=[ix[s]for s in ss]
defs(pathlib.Path(p['small_root'])/'native_word_evidence_v1.py',['new25']);defs(pathlib.Path(hp['word_text_screen'])/'word_text_worker_v1.py',['alignment']);defs(hp['normalization_source'],['expanded_words']);defs(hp['screen_worker'],['sequence_distance','phone_pair','phonetic_alignment']);defs(pathlib.Path(bp['protocol_root'])/'wavlm_head_worker_v1.py',['ridge','edits'])
assert sha(hp['cmu'])==hp['cmu_sha'];dictionary={}
for line in pathlib.Path(hp['cmu']).read_text().splitlines():
 a=line.split()
 if len(a)<2 or a[0].startswith(';;;'):continue
 word=re.sub(r'\(\d+\)$','',a[0]).lower();pron=tuple(re.sub(r'\d','',v)for v in a[1:])
 if pron and pron not in dictionary.get(word,[]):dictionary.setdefault(word,[]).append(pron)
public=read(bp['word_public']);contractions=read(hp['assets_protocol'])['contractions']
with np.load(B/'VOCAB_FULL_NATIVE48_ALL20239_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==train+dev;native48=z['features'][ii].copy();text=z['greedy_text'].tolist();ft=[text[i]for i in ii]
with np.load(W/'WORD_FULL_ALL20239_ADAPTED_GREEDY_TEXTS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==train+dev;rt=z['ref_text'].tolist();ht=z['human_text'].tolist();rt=[rt[i]for i in ii];ht=[ht[i]for i in ii]
assert sha(p['small_features_file'])==p['small_features_file_sha']
with np.load(p['small_features_file'],allow_pickle=False)as z:assert z['signals'].tolist()==ss;small=z['e1'].copy()
features={'small':small}
for name in ['e1','e2']:
 records=read(R/('PROJ_TDT_'+name.upper()+'_MICRO_RECORDS_V1.json'));assert set(records)==set(ss);rows=[]
 for j,s in enumerate(ss):
  x=new25(public[s]['words'],records[s],[ft[j],rt[j],ht[j]],contractions);rows.append(np.r_[x.mean(0),x.std(0)])
 features[name]=np.asarray(rows);assert features[name].shape==(5120,50)and np.isfinite(features[name]).all()
save('PROJ_TDT_MICRO_GLOBAL50_FEATURES_V1.npz',signals=np.asarray(ss),**features)
with np.load(bp['qwen_readouts']['44'],allow_pickle=False)as z:
 qs=z['FIT_signals'].tolist()+z['held_signals'].tolist();xx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qs)};q39=np.asarray([xx[qi[s]]for s in ss])
base=np.c_[q39,native48]
with np.load(bp['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;y=z['y'][ii].copy()
inputs=dict(base87=base,small137=np.c_[base,small],e1_137=np.c_[base,features['e1']],e2_137=np.c_[base,features['e3']]);pred={}
for name,X in inputs.items():
 q,norm=ridge(X,np.arange(N),y);pred[name]=q[N:];save('PROJ_TDT_'+name+'_COUNT_RIDGE_V1.npz',**norm)
assert sha(p['small_prediction_file'])==p['small_prediction_file_sha']
with np.load(p['small_prediction_file'],allow_pickle=False)as z:assert z['signals'].tolist()==held;assert np.max(abs(pred['small137']-z['e1_137']))<1e-8
for name in ['e1_137','e2_137']:
 for suffix,gamma in [('_quarter',.25),('_half',.5)]:pred[name+suffix]=gamma*pred[name]+(1-gamma)*pred['small137']
save('PROJ_TDT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(held),**pred);metrics={k:metric(v,y[N:])for k,v in pred.items()};priority=[name+suffix for name in ['e1_137','e2_137']for suffix in ['', '_quarter','_half']];selected=min(priority,key=lambda k:metrics[k]['RMSE']);gain=metrics['small137']['RMSE']-metrics[selected]['RMSE'];result=dict(status='native_TDT_projection_interface_adaptation_micro_complete',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>0),additional_global_ASR_steps=2520,prior_small_ASR_steps=768,ASR_routes=3,one_ASR_training_run=True,original_DEV_selected=False,seconds=time.monotonic()-start)
put('PROJ_TDT_SCREEN_RESULT_V1.json',result);put('PROJ_TDT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
