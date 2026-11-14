import os,sys,pathlib,json,hashlib,time,ast,math
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'DIRECT_PLAN_V1.json').read_text());H=pathlib.Path(p['source_full_root'])
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr,expit
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def save(n,**v):
 with(R/n).open('xb')as f:np.savez_compressed(f,**v)
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
def defs(f,names):
 nodes=[n for n in ast.parse(pathlib.Path(f).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f),'exec'),globals())
def build(hidden,seed):
 torch.manual_seed(88000+seed);net=nn.Sequential(nn.Linear(168,hidden),nn.GELU(),nn.Linear(hidden,1))
 with torch.no_grad():net[-1].weight.zero_();net[-1].bias.zero_()
 return net
def forward(net,x,anchor,legal,kind):
 z=net(x).squeeze(-1)
 if kind=='residual':z=z+torch.logit(anchor.clamp(1e-5,1-1e-5))[:,None]
 q=(torch.sigmoid(z)*legal).sum(1)/legal.sum(1)
 return q,z
start=time.monotonic()
for n,h in read(R/'DIRECT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
assert sha(p['source_full_plan'])==p['source_full_plan_sha'];hp=read(p['source_full_plan']);bp=read(hp['native_full_plan']);B=pathlib.Path(hp['native_full_root']);source=pathlib.Path(bp['protocol_root'])/'wavlm_head_worker_v1.py';assert sha(source)==bp['protocol_sha'];defs(source,['ridge'])
assert read(H/'HOMO_FULL_DRIVER_TERMINAL_V1.json')['actual_exit_code']==0
with np.load(bp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();FIT=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
N=len(FIT);ss=FIT+held;assert N==13340 and len(held)==2840 and set(ss)==set(train)and not set(ss)&set(dev)
save('DIRECT_PUBLIC_ROLES_BEFORE_FIT_V1.npz',FIT=np.asarray(FIT),held=np.asarray(held))
with np.load(H/'HOMO_FULL_ALL20239_LOCAL81_V1.npz',allow_pickle=False)as z:
 assert z['signals'].tolist()==train+dev;ii={s:i for i,s in enumerate(train)};ix=[ii[s]for s in ss];local=z['local'][ix].copy();legal=z['legal'][ix].copy()
valid=local[:N][legal[:N]];lmu=valid.mean(0);lsd=np.maximum(valid.std(0),1e-4);ZL=np.clip((local-lmu)/lsd,-6,6);save('DIRECT_FIT_WORD_NORM_V1.npz',word_mean=lmu,word_std=lsd)
with np.load(B/'VOCAB_FULL_NATIVE48_ALL20239_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==train+dev;native=z['features'][ix].copy()
with np.load(bp['qwen_readouts']['44'],allow_pickle=False)as z:
 qq=z['FIT_signals'].tolist()+z['held_signals'].tolist();xx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qq)};q39=np.asarray([xx[qi[s]]for s in ss])
with np.load(bp['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;y=z['y'][ix].copy()
with np.load(hp['word_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;wy=z['word_targets'][ix[:N]].copy();known=z['word_target_mask'][ix[:N]].copy();assert np.all(known<=legal[:N])
raw=np.c_[q39,native];pred,nm=ridge(raw,np.arange(N),y);save('DIRECT_FIT_RIDGE87_AND_NORM_V1.npz',**nm);ZG=np.clip((np.where(np.isfinite(raw),raw,nm['median'])-nm['mean'])/nm['std'],-6,6);anchor=pred.copy();public=read(bp['word_public']);fold=np.asarray([int(digest(['native87-nonlinear-oof-v1',' '.join(public[s]['words'])])[:8],16)%5 for s in FIT]);save('DIRECT_FIT_PUBLIC_FOLDS_V1.npz',signals=np.asarray(FIT),fold=fold)
for f in range(5):
 q,st=ridge(raw,np.where(fold!=f)[0],y);ii=np.where(fold==f)[0];anchor[ii]=q[ii];save('DIRECT_FIT_RIDGE_OOF_FOLD%d_V1.npz'%f,**st)
XX=np.concatenate([np.repeat(ZG[:,None,:],12,axis=1),ZL],axis=2);XT=torch.tensor(XX,dtype=torch.float32);AT=torch.tensor(anchor,dtype=torch.float32);LT=torch.tensor(legal,dtype=torch.float32);YT=torch.tensor(y,dtype=torch.float32);WY=torch.tensor(wy,dtype=torch.float32);KM=torch.tensor(known);preds={};models=[];proof=[]
for mode,cfg in p['models'].items():
 net=build(cfg['hidden'],44);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(99044);steps=0
 for epoch in range(8):
  order=torch.randperm(N,generator=rng).numpy()
  for pos in range(0,N,128):
   ii=order[pos:pos+128];q,z=forward(net,XT[ii],AT[ii],LT[ii],cfg['kind']);loss=(q-YT[ii]).square().mean()
   if KM[ii].any():loss=loss+cfg['word_BCE']*F.binary_cross_entropy_with_logits(z[KM[ii]],WY[ii][KM[ii]])
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
 assert steps==840;file='DIRECT_'+mode+'_MODEL_SEED44_V1.pt'
 with(R/file).open('xb')as f:torch.save(dict(weights=net.state_dict(),optimizer=opt.state_dict(),steps=steps,seed=44,configuration=cfg,FIT_ID_SHA=digest(FIT),known_FIT_word_labels=int(known.sum())),f)
 with torch.no_grad():q,z=forward(net,XT[N:],AT[N:],LT[N:],cfg['kind']);q=q.double().numpy()
 state={k:v.numpy().astype(np.float64)for k,v in net.state_dict().items()};hh=XT[N:].double().numpy()@state['0.weight'].T+state['0.bias'];delta=((hh*ndtr(hh))@state['2.weight'].T+state['2.bias'])[...,0]
 if cfg['kind']=='residual':aa=AT[N:].double().numpy().clip(1e-5,1-1e-5);delta=delta+np.log(aa/(1-aa))[:,None]
 pp=expit(delta);oq=(pp*legal[N:]).sum(1)/legal[N:].sum(1);gap=float(np.max(abs(q-oq)));assert gap<2e-6;preds[mode]=q;proof.append(dict(mode=mode,all2840_NumPy_maxgap=gap));models.append(dict(mode=mode,steps=steps,checkpoint_sha=sha(R/file)))
for mode in ['direct32','direct64']:
 for tag,gamma in [('blend25',.25),('blend50',.5)]:preds[mode+'_'+tag]=gamma*preds[mode]+(1-gamma)*preds['baseline']
save('DIRECT_ALL_LARGER_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(held),**preds)
metrics={k:metric(v,y[N:])for k,v in preds.items()};priority=['word30','direct32','direct64','direct32_blend25','direct32_blend50','direct64_blend25','direct64_blend50'];selected=min(priority,key=lambda n:metrics[n]['RMSE']);gain=metrics['baseline']['RMSE']-metrics[selected]['RMSE'];result=dict(status='larger_TRAINheld_direct_word_screen_complete',role='FIT13340_historicalTRAINheld2840_NOT_original4059',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>0),new_score_updates=3360,models=models,independent_NumPy=proof,formal_DEV_gold=False,prior_small_reg_gate_failed_preserved=True,seconds=time.monotonic()-start)
put('DIRECT_LARGER_SCREEN_RESULT_V1.json',result);put('DIRECT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
