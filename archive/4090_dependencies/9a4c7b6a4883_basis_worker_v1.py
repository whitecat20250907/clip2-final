"""Frozen cached Parakeet/Granite frames, own dual-phone heads; TRAIN-only single-seed screen."""
import os,sys,pathlib,json,hashlib,time,ast,math
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'BASIS_PLAN_V1.json').read_text())
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def emit(n,v):
 with (R/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def stash(n,**v):
 with (R/n).open('xb') as f:np.savez_compressed(f,**v)
def progress(**v):
 q=R/'progress.next.json';q.write_text(json.dumps(v));os.replace(q,R/'progress.json')
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
start=time.monotonic()
for n,h in read(R/'BASIS_BINDING_V1.json')['files'].items():assert sha(R/n)==h,n
for k in ['roles','micro_roles','count_targets','public','auxiliary','qwen_readout']:assert sha(p[k])==p[k+'_sha'],k
source=pathlib.Path(p['previous_protocol_root'])/'wavlm_head_worker_v1.py';assert sha(source)==p['previous_protocol_sha']
names=['minimum','greedy','edits','scalar16','dual35','ridge'];nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'))
with np.load(p['roles'],allow_pickle=False) as z:train=z['strict_TRAIN_ids'].tolist();FIT=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
assert len(train)==16180 and digest(train)==p['TRAIN_ID_SHA'] and len(dev)==4059 and digest(dev)==p['DEV_ID_SHA']
with np.load(p['micro_roles'],allow_pickle=False) as z:countfit=z['FIT'].tolist();countheld=z['held'].tolist()
assert len(countfit)==4096 and len(countheld)==1024 and set(countfit)<=set(FIT) and set(countheld)<=set(held) and not set(countfit)&set(countheld)
signals=countfit+countheld;aux=read(p['auxiliary']);public=read(p['public']);assert len(aux)==10077 and set(aux)<=set(FIT)
indices={};frames={}
for family,entry in p['caches'].items():
 C=pathlib.Path(entry['root']);assert sha(C/'index.json')==entry['index_sha'] and sha(entry['source'])==entry['source_sha']
 index=read(C/'index.json');assert index['model']==entry['model'] and index['dim']==1024 and index['dtype']=='float16' and index['n']==20297
 assert read(C/'extraction_summary.json')['status']=='complete' and (C/'features.f16').stat().st_size==index['total_tokens']*1024*2
 assert len(index['signal_ids'])==len(set(index['signal_ids']))==len(index['offsets'])==len(index['lengths'])==20297
 assert set(train+dev)<=set(index['signal_ids']);indices[family]=index;frames[family]={s:int(n) for s,n in zip(index['signal_ids'],index['lengths'])}
auxids=[s for s in sorted(aux) if all(minimum(aux[s][key])<=2*frames[family][s] for family in p['caches'] for key in ['reference','human'])]
assert auxids and set(auxids)<=set(FIT)
stash('BASIS_PUBLIC_ROLES_BEFORE_TRAINING_V1.npz',countFIT=np.asarray(countfit),countHeld=np.asarray(countheld),auxiliary_training=np.asarray(auxids))
emit('BASIS_SUPERVISION_SCOPE_BEFORE_TRAINING_V1.json',dict(aux_original=10077,aux_common_feasible=len(auxids),aux_unfeasible_count=10077-len(auxids),aux_TRAIN_ID_SHA=digest(auxids),countFIT_rows=4096,countHeld_rows=1024,no_count_rows_removed=True,global_seed=44,epochs=3,updates_per_family=3*math.ceil(len(auxids)/32),formal_DEV_gold_or_features_used=False,interpolation='perclipCPUdouble2xlogitsbeforelogsoftmaxANDbatchpad'))
with np.load(p['count_targets'],allow_pickle=False) as z:assert z['signals'].tolist()==train;yt=z['y'].copy()
ix={s:i for i,s in enumerate(train)};y=np.asarray([yt[ix[s]] for s in signals]);fi=np.arange(4096);he=np.arange(4096,5120)
with np.load(p['qwen_readout'],allow_pickle=False) as z:
 qids=z['FIT_signals'].tolist()+z['held_signals'].tolist();qraw=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qids)};q39=np.asarray([qraw[qi[s]] for s in signals])
qpred,norm=ridge(q39,fi,y);pred={'Qwen39_baseline':qpred[he]};stash('BASIS_QWEN39_CONTROL_READOUT_V1.npz',**norm)
assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=2*1024**3;torch.cuda.set_per_process_memory_fraction(1024**3/total)
def make_heads():
 hs={}
 for name,seed in [('ref',80044),('response',81044)]:
  torch.manual_seed(seed);h=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU(),nn.Linear(128,40))
  with torch.no_grad():h[-1].bias.zero_();h[-1].bias[0]=math.log(39.)
  hs[name]=h
 out=nn.ModuleDict(hs).to('cuda:0');assert sum(v.numel() for v in out.parameters())==276816
 return out
def batch_lp(hs,ss,bank):
 lens=[len(bank[s]) for s in ss];xx=torch.from_numpy(np.concatenate([bank[s] for s in ss],axis=0)).to('cuda:0');assert xx.shape[1]==1024 and not xx.requires_grad
 raw=torch.stack([hs[name](xx) for name in ['ref','response']],dim=1).double().cpu();pieces=torch.split(raw,lens,dim=0);out=[]
 for a,n in zip(pieces,lens):
  lp=torch.stack([F.interpolate(a[:,j].T[None],scale_factor=2,mode='linear',align_corners=False)[0].T.log_softmax(-1) for j in [0,1]],dim=1)
  assert lp.shape==(2*n,2,40) and torch.isfinite(lp).all()
  out.append(lp)
 return out
def normalized_ctc_per_row(lps,ss):
 rows=[]
 for j,key in enumerate(['reference','human']):
  seq=[aux[s][key] for s in ss];target=torch.tensor([v for r in seq for v in r],dtype=torch.long);tl=torch.tensor([len(r) for r in seq],dtype=torch.long);ll=torch.tensor([len(v) for v in lps],dtype=torch.long)
  pp=nn.utils.rnn.pad_sequence([v[:,j] for v in lps],batch_first=False,padding_value=0.)
  loss=F.ctc_loss(pp,target,ll,tl,blank=0,reduction='none',zero_infinity=False)/tl.clamp_min(1)
  assert loss.shape==(len(ss),) and torch.isfinite(loss).all();rows.append(loss)
 return .5*(rows[0]+rows[1])
models=[];qualifications=[];total_updates=0
for family,entry in p['caches'].items():
 index=indices[family];ci={s:i for i,s in enumerate(index['signal_ids'])};wanted=set(auxids+signals);assert wanted<=set(train)
 mm=np.memmap(pathlib.Path(entry['root'])/'features.f16',dtype=np.float16,mode='r',shape=(index['total_tokens'],1024));bank={}
 for pos,s in enumerate(sorted(wanted,key=lambda s:ci[s])):
  j=ci[s];off=index['offsets'][j];n=index['lengths'][j];v=np.array(mm[off:off+n],dtype=np.float32,copy=True);assert v.shape==(n,1024) and n>0 and np.isfinite(v).all();bank[s]=v
  if pos%1024==0:progress(stage='public_TRAIN_frame_RAM',family=family,processed=pos+1,total=len(wanted),seconds=time.monotonic()-start);print('BASIS_RAM',family,pos+1,len(wanted),flush=True)
 del mm
 hs=make_heads();qual_ids=auxids[:2];lps=batch_lp(hs,qual_ids,bank);losses=normalized_ctc_per_row(lps,qual_ids);individual=[]
 for lp,s in zip(lps,qual_ids):
  parts=[]
  for j,key in enumerate(['reference','human']):
   target=torch.tensor(aux[s][key],dtype=torch.long);parts.append(F.ctc_loss(lp[:,j][:,None],target,torch.tensor([len(lp)]),torch.tensor([len(target)]),blank=0,reduction='none',zero_infinity=False)[0]/max(1,len(target)))
  individual.append(.5*(parts[0]+parts[1]))
 cgap=float(torch.max(abs(losses-torch.stack(individual))).detach());assert cgap<1e-8
 with torch.no_grad():sgap=max(float(torch.max(abs(batch_lp(hs,[s],bank)[0]-lp)).detach()) for s,lp in zip(qual_ids,lps))
 assert sgap<1e-5
 grads=torch.autograd.grad(losses.mean(),tuple(hs.parameters()),allow_unused=True);gn=sum(float(g.norm()) for g in grads if g is not None);assert gn>0 and np.isfinite(gn) and all(g is None or torch.isfinite(g).all() for g in grads)
 q=dict(family=family,clips=qual_ids,batch_vs_perclip_CTC_maxgap=cgap,packed_vs_single_LP_maxgap=sgap,head_gradient_norm=gn,trainable_head_parameters=276816,encoder_model_instantiated=False,new_optimizer_updates=0);qualifications.append(q);emit('BASIS_'+family.upper()+'_QUALIFICATION_V1.json',q)
 del lps,losses,individual,grads
 opt=torch.optim.AdamW(hs.parameters(),lr=.001,weight_decay=.001,foreach=False);gen=torch.Generator().manual_seed(60044);steps=0;loss_values=[]
 for epoch in range(3):
  order=torch.randperm(len(auxids),generator=gen).tolist()
  for pos in range(0,len(order),32):
   ss=[auxids[j] for j in order[pos:pos+32]];lps=batch_lp(hs,ss,bank);loss=normalized_ctc_per_row(lps,ss).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(hs.parameters(),1.,error_if_nonfinite=True,foreach=False);opt.step();steps+=1;loss_values.append(float(loss.detach()))
   if steps%64==0:progress(stage='own_cached_basis_phone_head_training',family=family,steps=steps,total=3*math.ceil(len(auxids)/32),epoch=epoch+1,seconds=time.monotonic()-start);print('BASIS_UPDATES',family,steps,float(np.mean(loss_values[-64:])),flush=True)
 assert steps==3*math.ceil(len(auxids)/32);total_updates+=steps
 cp=R/('BASIS_'+family.upper()+'_HEADS_SEED44_V1.pt')
 with cp.open('xb') as f:torch.save(dict(heads={k:v.detach().cpu() for k,v in hs.state_dict().items()},optimizer=opt.state_dict(),steps=steps,global_seed=44,auxiliary_training_ids=auxids,auxiliary_ID_SHA=digest(auxids),source_cache_root=entry['root'],source_index_sha=entry['index_sha'],source_model=entry['model'],source_auxiliary_sha=p['auxiliary_sha'],interpolation='2x_perclip_CPUdouble_logits_then_logsoftmax',trainable_head_parameters=276816,frozen_cached_encoder=True),f)
 hs.eval();features=np.empty((5120,35),np.float64)
 with torch.no_grad():
  for pos in range(0,len(signals),32):
   ss=signals[pos:pos+32];lps=batch_lp(hs,ss,bank)
   for j,(s,lp) in enumerate(zip(ss,lps)):features[pos+j]=dual35(lp[:,0].numpy(),lp[:,1].numpy(),public[s])
   if pos%256==0:progress(stage='TRAIN_count_features35',family=family,processed=min(pos+32,len(signals)),total=5120,seconds=time.monotonic()-start);print('BASIS_FEATURES',family,min(pos+32,len(signals)),flush=True)
 stash('BASIS_'+family.upper()+'_MICRO_FEATURES35_V1.npz',signals=np.asarray(signals),features=features)
 for key,X in [(family+'35',features),('qwen_'+family+'74',np.c_[q39,features])]:
  qv,nm=ridge(X,fi,y);pred[key]=qv[he];stash('BASIS_'+key+'_READOUT_V1.npz',**nm)
 models.append(dict(family=family,steps=steps,auxiliary_rows=len(auxids),checkpoint_sha=sha(cp),encoder_updates=0,mean_training_loss=float(np.mean(loss_values)),index_sha=entry['index_sha']))
 del opt,hs,bank
stash('BASIS_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(countheld),**pred)
ms={k:metric(v,y[he]) for k,v in pred.items()};selected=min((k for k in ms if k!='Qwen39_baseline'),key=lambda k:ms[k]['RMSE']);gain=ms['Qwen39_baseline']['RMSE']-ms[selected]['RMSE']
family=next(k for k in p['caches'] if k in selected)
emit('BASIS_FAST_SCREEN_RESULT_V1.json',dict(status='public_cached_ASR_basis_own_dualphone_TRAIN_screen_complete',scope='countFIT4096_held1024_and_ASRauxFIT_only_not_original4059',metrics=ms,models=models,qualifications=qualifications,selected=selected,selected_family=family,gain_vs_Qwen39=gain,promising=gain>.05,total_new_head_updates=total_updates,encoder_updates=0,global_seed=44,not_two_final_seeds=True,formal_DEV_gold_or_features_used=False,seconds=time.monotonic()-start,limits=p['limits']))
print('BASIS_SCREEN',ms,selected,gain,flush=True)
