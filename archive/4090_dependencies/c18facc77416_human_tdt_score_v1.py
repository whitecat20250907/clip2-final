import os,sys,pathlib,json,hashlib,time,ast,math,re
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'HUMAN_TDT_PLAN_V1.json').read_text())
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
for n,h in read(R/'HUMAN_TDT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
assert read(R/'HUMAN_TDT_ASR_TRAIN_TERMINAL_V1.json')['actual_exit_code']==0;seal=read(R/'HUMAN_TDT_ADAPTERS_SEALED_V1.json');assert seal['actual_optimizer_steps']==768 and sha(R/seal['checkpoint'])==seal['checkpoint_sha']
hp=read(p['source_full_plan']);bp=read(hp['native_full_plan']);B=pathlib.Path(hp['native_full_root']);W=pathlib.Path(hp['word_full_root']);S=pathlib.Path(p['native_root'])/'readout_screen_v1'
with np.load(bp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
with np.load(p['micro_roles'],allow_pickle=False)as z:fit=z['FIT'].tolist();held=z['held'].tolist()
ss=fit+held;N=len(fit);assert N==4096 and len(held)==1024 and not set(ss)&set(dev);ix={s:i for i,s in enumerate(train)};ii=[ix[s]for s in ss]
defs(S/'tdt_readout_worker_v1.py',['new25']);defs(pathlib.Path(hp['word_text_screen'])/'word_text_worker_v1.py',['alignment']);defs(hp['normalization_source'],['expanded_words']);defs(hp['screen_worker'],['sequence_distance','phone_pair','phonetic_alignment']);defs(pathlib.Path(bp['protocol_root'])/'wavlm_head_worker_v1.py',['ridge','edits'])
assert sha(hp['cmu'])==hp['cmu_sha'];dictionary={}
for line in pathlib.Path(hp['cmu']).read_text().splitlines():
 a=line.split()
 if len(a)<2 or a[0].startswith(';;;'):continue
 word=re.sub(r'\(\d+\)$','',a[0]).lower();pron=tuple(re.sub(r'\d','',v)for v in a[1:])
 if pron and pron not in dictionary.get(word,[]):dictionary.setdefault(word,[]).append(pron)
public=read(bp['word_public']);contractions=read(hp['assets_protocol'])['contractions']
with np.load(B/'VOCAB_FULL_NATIVE48_ALL20239_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==train+dev;native48=z['features'][ii].copy();text=z['greedy_text'].tolist();ft=[text[i]for i in ii]
with np.load(W/'WORD_FULL_ALL20239_ADAPTED_GREEDY_TEXTS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==train+dev;rt=z['ref_text'].tolist();ht=z['human_text'].tolist();rt=[rt[i]for i in ii];ht=[ht[i]for i in ii]
assert sha(p['frozen25_features_file'])==p['frozen25_features_file_sha']
with np.load(p['frozen25_features_file'],allow_pickle=False)as z:fi={s:i for i,s in enumerate(z['signals'].tolist())};local=z['local25'].copy()
features={}
for name in ['frozen','ref','human']:
 records=None if name=='frozen'else read(R/('HUMAN_TDT_'+name.upper()+'_MICRO_RECORDS_V1.json'));rows=[]
 if records is not None:assert set(records)==set(ss)
 for j,s in enumerate(ss):
  n=len(public[s]['words']);x=local[fi[s],:n]if records is None else new25(public[s]['words'],records[s],[ft[j],rt[j],ht[j]],contractions);rows.append(np.r_[x.mean(0),x.std(0)])
 features[name]=np.asarray(rows);assert features[name].shape==(5120,50)and np.isfinite(features[name]).all()
save('HUMAN_TDT_MICRO_GLOBAL50_FEATURES_V1.npz',signals=np.asarray(ss),**features)
with np.load(bp['qwen_readouts']['44'],allow_pickle=False)as z:
 qs=z['FIT_signals'].tolist()+z['held_signals'].tolist();xx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qs)};q39=np.asarray([xx[qi[s]]for s in ss])
base=np.c_[q39,native48]
with np.load(bp['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;y=z['y'][ii].copy()
f=features['frozen'];r=features['ref'];h=features['human'];inputs=dict(base87=base,frozen137=np.c_[base,f],frozen187=np.c_[base,f,f],ref137=np.c_[base,r],human137=np.c_[base,h],dual187=np.c_[base,r,h]);pred={}
for name,X in inputs.items():
 q,norm=ridge(X,np.arange(N),y);pred[name]=q[N:];save('HUMAN_TDT_'+name+'_COUNT_RIDGE_V1.npz',**norm)
matched=dict(ref137='frozen137',human137='frozen137',dual187='frozen187')
for name,control in matched.items():pred[name+'_half']=.5*pred[name]+.5*pred[control]
save('HUMAN_TDT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(held),**pred);metrics={k:metric(v,y[N:])for k,v in pred.items()};assert abs(metrics['base87']['RMSE']-28.14937933)<1e-6;priority=[name+tag for name in matched for tag in ['', '_half']];eligible=[name for name in priority if metrics[name]['RMSE']<metrics[matched[name.replace('_half','')]]['RMSE']];selected=min(eligible,key=lambda k:metrics[k]['RMSE'])if eligible else None;control=matched[selected.replace('_half','')]if selected else None;gain=metrics[control]['RMSE']-metrics[selected]['RMSE']if selected else None
result=dict(status='human_native_TDT_LoRA_micro_complete',metrics=metrics,selected=selected,matched_control=control,gain=gain,promising=bool(eligible),actual_ASR_optimizer_steps=768,ASR_heads=['ref','human'],shared_adapter_seed44=True,original_DEV_selected=False,seconds=time.monotonic()-start)
put('HUMAN_TDT_SCREEN_RESULT_V1.json',result);put('HUMAN_TDT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
