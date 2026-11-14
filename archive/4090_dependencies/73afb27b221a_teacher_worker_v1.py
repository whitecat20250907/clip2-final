import os,sys,pathlib,json,hashlib,time,ast,math
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'TEACHER_PLAN_V1.json').read_text())
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','TORCH_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME','TMPDIR']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def save(n,**v):
 with(R/n).open('xb')as f:np.savez_compressed(f,**v)
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
start=time.monotonic()
for n,h in read(R/'TEACHER_BINDING_V1.json')['files'].items():assert sha(R/n)==h
NAT=pathlib.Path(p['native_root']);S=pathlib.Path(p['readout_root']);D=pathlib.Path(p['larger_root']);H=pathlib.Path(p['source_full_root']);hp=read(H/'HOMO_FULL_PLAN_V1.json');bp=read(hp['native_full_plan']);B=pathlib.Path(hp['native_full_root']);np_=read(p['native_plan'])
assert sha(p['micro_roles'])==p['micro_roles_sha']
with np.load(p['micro_roles'],allow_pickle=False)as z:fit=z['FIT'].tolist();held=z['held'].tolist()
with np.load(bp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();dev=z['formal_DEV_ids'].tolist();bigfit=z['FIT_ids'].tolist();bigheld=z['historical_held_ids'].tolist()
assert len(fit)==4096 and len(held)==1024 and set(fit)<=set(bigfit)and set(held)<=set(bigheld)and not set(fit+held)&set(dev);ss=fit+held;N=len(fit);ix={s:i for i,s in enumerate(train)};ii=[ix[s]for s in ss]
assert sha(NAT/'native_tdt_core_v1.py')==p['native_core_sha'];sys.path.insert(0,str(NAT))
from native_tdt_core_v1 import NativeTDT
from teacher_support_core_v1 import extract
assert sha(S/'TDT_READOUT_direct193_MODEL_SEED44_V1.pt')==p['warm_checkpoint_sha']
C=pathlib.Path(np_['cache']['root']);assert sha(C/'index.json')==np_['cache']['index_sha'];index=read(C/'index.json');ci={s:i for i,s in enumerate(index['signal_ids'])};mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
def frames(s):
 k=ci[s];off=index['offsets'][k];n=index['lengths'][k];return np.asarray(mm[off:off+n],dtype=np.float32).copy()
public=read(bp['word_public']);free,total=torch.cuda.mem_get_info();assert free>3*1024**3;torch.cuda.set_per_process_memory_fraction(min(.9,2*1024**3/total));engine=NativeTDT(np_);assert engine.weight_sha==p['native_weights_sha']
extra=np.zeros((len(ss),12,32));qualification=[]
with torch.inference_mode():
 for k,s in enumerate(ss):
  value,ret=extract(engine,frames(s),public[s]['words']);extra[k,:len(value)]=value
  if k<2:
   ref,rr=extract(engine,frames(s),public[s]['words'],dense=True);gap=max(float(np.max(abs(a-b)))for aa,bb in zip(ret,rr)for a,b in zip(aa,bb));wgap=float(np.max(abs(value-ref)));assert max(gap,wgap)<2e-4;qualification.append(dict(signal=s,retainedLP_maxgap=gap,word_feature_maxgap=wgap))
  if (k+1)%256==0:print(json.dumps(dict(phase='teacher_support',completed=k+1,total=len(ss),seconds=time.monotonic()-start)),flush=True)
put('TEACHER_QUALIFICATION_V1.json',dict(tests=qualification,weight_sha=engine.weight_sha,no_encoder_execution=True));del engine;torch.cuda.empty_cache()
save('TEACHER_LOCAL32_MICRO_V1.npz',signals=np.asarray(ss),local32=extra)
with np.load(H/'HOMO_FULL_ALL20239_LOCAL81_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==train+dev;oldlocal=z['local'][ii].copy();legal=z['legal'][ii].copy()
with np.load(S/'TDT_NEW_LOCAL25_TRAIN_V1.npz',allow_pickle=False)as z:
 ei={s:i for i,s in enumerate(z['signals'].tolist())};ee=z['local25'].copy();old25=ee[[ei[s]for s in ss]]
with np.load(S/'TDT_READOUT_NEW25_FIT_NORM_V1.npz',allow_pickle=False)as z:ZE=np.clip((old25-z['mean'])/z['std'],-6,6)
with np.load(D/'DIRECT_FIT_WORD_NORM_V1.npz',allow_pickle=False)as z:ZL=np.clip((oldlocal-z['word_mean'])/z['word_std'],-6,6)
with np.load(D/'DIRECT_FIT_RIDGE87_AND_NORM_V1.npz',allow_pickle=False)as z:nm={k:z[k].copy()for k in z.files}
with np.load(B/'VOCAB_FULL_NATIVE48_ALL20239_FEATURES_V1.npz',allow_pickle=False)as z:native48=z['features'][ii].copy()
with np.load(bp['qwen_readouts']['44'],allow_pickle=False)as z:
 qs=z['FIT_signals'].tolist()+z['held_signals'].tolist();xx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qs)};q39=np.asarray([xx[qi[s]]for s in ss])
raw=np.c_[q39,native48];ZG=np.clip((np.where(np.isfinite(raw),raw,nm['median'])-nm['mean'])/nm['std'],-6,6);oldX=np.concatenate([np.repeat(ZG[:,None,:],12,axis=1),ZL,ZE],axis=2);assert oldX.shape==(5120,12,193)
valid=extra[:N][legal[:N]];mu=valid.mean(0);sd=np.maximum(valid.std(0),1e-4);XX=np.concatenate([oldX,np.clip((extra-mu)/sd,-6,6)],axis=2);save('TEACHER_FIT32_NORM_V1.npz',mean=mu,std=sd)
with np.load(bp['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;y=z['y'][ii].copy()
with np.load(hp['word_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;wy=z['word_targets'][ii[:N]].copy();known=z['word_target_mask'][ii[:N]].copy()
XT=torch.tensor(XX,dtype=torch.float32);LT=torch.tensor(legal,dtype=torch.float32);YT=torch.tensor(y,dtype=torch.float32);WY=torch.tensor(wy,dtype=torch.float32);KM=torch.tensor(known)
cp=torch.load(S/'TDT_READOUT_direct193_MODEL_SEED44_V1.pt',map_location='cpu',weights_only=False);assert cp['steps']==840
def predict(net,x):
 z=net(x).squeeze(-1);return (z.sigmoid()*LT[N:]).sum(1)/LT[N:].sum(1)
warm=nn.Sequential(nn.Linear(193,32),nn.GELU(),nn.Linear(32,1));warm.load_state_dict(cp['weights'])
with torch.no_grad():base=predict(warm,XT[N:,:,:193]).double().numpy()
with np.load(S/'TDT_READOUT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',allow_pickle=False)as z:
 si={s:i for i,s in enumerate(z['signals'].tolist())};saved=z['direct193'].copy()[[si[s]for s in held]]
assert np.max(abs(saved-base))<1e-8;pred={'warm193':base};models=[]
for mode,dim in [('teacher225',225),('continue193',193)]:
 net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1))
 with torch.no_grad():
  net[0].weight.zero_();net[0].weight[:,:193].copy_(warm[0].weight);net[0].bias.copy_(warm[0].bias);net[2].weight.copy_(warm[2].weight);net[2].bias.copy_(warm[2].bias)
 with torch.no_grad():assert torch.max(abs(predict(net,XT[N:,:,:dim])-torch.tensor(base,dtype=torch.float32)))<2e-7
 opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(99144);steps=0
 for epoch in range(4):
  order=torch.randperm(N,generator=rng).numpy()
  for pos in range(0,N,128):
   jj=order[pos:pos+128];z=net(XT[jj,:,:dim]).squeeze(-1);q=(z.sigmoid()*LT[jj]).sum(1)/LT[jj].sum(1);loss=(q-YT[jj]).square().mean()
   if KM[jj].any():loss=loss+.3*F.binary_cross_entropy_with_logits(z[KM[jj]],WY[jj][KM[jj]])
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
 assert steps==128
 with(R/('TEACHER_'+mode+'_MODEL_SEED44_V1.pt')).open('xb')as f:torch.save(dict(weights=net.state_dict(),optimizer=opt.state_dict(),steps=steps,previous_steps=840,dimension=dim),f)
 with torch.no_grad():pred[mode]=predict(net,XT[N:,:,:dim]).double().numpy()
 models.append(dict(mode=mode,new_steps=steps,checkpoint_sha=sha(R/('TEACHER_'+mode+'_MODEL_SEED44_V1.pt'))))
save('TEACHER_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(held),**pred)
metrics={k:metric(v,y[N:])for k,v in pred.items()};selected=min(['teacher225','continue193'],key=lambda k:metrics[k]['RMSE']);gain=metrics['warm193']['RMSE']-metrics[selected]['RMSE'];result=dict(status='teacher_support_pilot_complete',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>0),teacher_vs_control=metrics['continue193']['RMSE']-metrics['teacher225']['RMSE'],models=models,new_score_updates=256,new_ASR_updates=0,original_DEV_gold=False,seconds=time.monotonic()-start)
put('TEACHER_SCREEN_RESULT_V1.json',result);put('TEACHER_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
