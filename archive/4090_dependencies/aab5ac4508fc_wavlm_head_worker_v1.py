"""Own dual phone CTC heads on qualified public WavLM50Hz features; no peer fitted artifacts."""
import os,sys,pathlib,json,hashlib,time,math
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'WAVLM_HEAD_PLAN_V1.json').read_text())
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(R/'runtime/CUDA_CACHE'))
(R/'runtime/CUDA_CACHE').mkdir(parents=True,exist_ok=True)
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):
 h=hashlib.sha256()
 with pathlib.Path(f).open('rb') as v:
  for b in iter(lambda:v.read(4194304),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v):
 with (R/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def save(n,**v):
 with (R/n).open('xb') as f:np.savez_compressed(f,**v)
def progress(**v):
 q=R/'progress.next.json';q.write_text(json.dumps(v));os.replace(q,R/'progress.json')
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
def minimum(seq):return len(seq)+sum(a==b for a,b in zip(seq,seq[1:]))
def greedy(lp):
 ids=lp.argmax(1).tolist();return [v for i,v in enumerate(ids) if v and (i==0 or v!=ids[i-1])]
def edits(ref,hyp):
 nr,nh=len(ref),len(hyp);d=np.empty((nr+1,nh+1),np.int64);d[:,0]=np.arange(nr+1);d[0]=np.arange(nh+1)
 for i in range(1,nr+1):
  for j in range(1,nh+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
 i,j=nr,nh;s=de=ins=0
 while i or j:
  if i and j and ref[i-1]==hyp[j-1] and d[i,j]==d[i-1,j-1]:i-=1;j-=1
  elif i and j and d[i,j]==d[i-1,j-1]+1:s+=1;i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:de+=1;i-=1
  else:ins+=1;j-=1
 assert s+de+ins==d[-1,-1];return s,de,ins
def scalar16(lp,record):
 prob=np.exp(lp);dec=greedy(lp);ref=record['reference'];n=record['n_words'];sev=record['severity']
 a=np.full(16,np.nan,np.float64)
 a[5]=float(-(prob*lp).sum(1).mean()/np.log(40));a[6]=float(prob[:,0].mean());a[7]=float(prob.max(1).mean())
 a[9]=np.log1p(len(lp));a[11]=n;a[12]=1.;a[13]=sev==1;a[14]=sev==2;a[15]=1/n
 if ref:
  nr=len(ref);s,de,ins=edits(ref,dec);a[1]=np.clip(1-(s+de+ins)/nr,0,1);a[2:5]=[s/nr,de/nr,ins/nr]
  a[8]=np.log1p(len(dec))-np.log1p(nr);a[10]=np.log1p(nr)
  if minimum(ref)<=len(lp):
   tt=torch.tensor(ref,dtype=torch.long);tl=torch.tensor([nr]);length=torch.tensor([len(lp)])
   a[0]=float(F.ctc_loss(torch.from_numpy(lp)[:,None,:],tt,length,tl,blank=0,reduction='none',zero_infinity=False)[0])/nr;a[12]=0.
 return a
def dual35(a,b,record):
 pa,pb=np.exp(a),np.exp(b);mid=np.logaddexp(a,b)-np.log(2.)
 js=float(.5*((pa*(a-mid)).sum(1)+(pb*(b-mid)).sum(1)).mean()/np.log(2.))
 blank=float(abs(pa[:,0]-pb[:,0]).mean());ga,gb=greedy(a),greedy(b);s,d,i=edits(ga,gb)
 v=np.r_[scalar16(a,record),scalar16(b,record),js,blank,(s+d+i)/max(1,len(ga),len(gb))]
 assert v.shape==(35,) and np.isfinite(v[[5,6,7,9,11,12,13,14,15,21,22,23,25,27,28,29,30,31,32,33,34]]).all();return v
def ridge(x,fitrows,y):
 tr=x[fitrows];med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in tr.T]);xx=np.where(np.isfinite(x),x,med)
 mu=xx[fitrows].mean(0);sd=np.maximum(xx[fitrows].std(0),1e-4);a=np.c_[np.ones(len(xx)),np.clip((xx-mu)/sd,-6,6)]
 reg=np.diag(np.r_[0.,np.full(x.shape[1],10.)]);coef=np.linalg.solve(a[fitrows].T@a[fitrows]+reg,a[fitrows].T@y[fitrows])
 return np.clip(a@coef,0,1),dict(median=med,mean=mu,std=sd,coef=coef)
start=time.monotonic()
for n,h in read(R/'WAVLM_HEAD_BINDING_V1.json')['files'].items():assert sha(R/n)==h,n
Q=pathlib.Path(p['qualification_root']);qual=read(Q/'WAVLM_PUBLIC_CACHE_QUALIFICATION_V1.json')
assert qual['qualified'] and read(Q/'WAVLM_QUAL_DRIVER_TERMINAL_V1.json')['actual_exit_code']==0
with np.load(p['roles'],allow_pickle=False) as z:train=z['strict_TRAIN_ids'].tolist();fit=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist()
assert len(train)==16180 and len(fit)==13340 and len(held)==2840 and digest(train)==p['TRAIN_ID_SHA'] and not set(fit)&set(held)
look={s:i for i,s in enumerate(train)};fi=np.asarray([look[s] for s in fit]);he=np.asarray([look[s] for s in held])
assert sha(p['auxiliary'])==p['auxiliary_sha'];aux=read(p['auxiliary']);assert len(aux)==10077 and set(aux)<=set(fit)
public=read(p['public']);C=pathlib.Path(p['cache']);assert sha(C/'index.json')==p['index_sha'];index=read(C/'index.json')
bank=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],768));cacheix={s:i for i,s in enumerate(index['signal_ids'])}
assert set(train)<=set(cacheix);off=index['offsets'];lengths=index['lengths']
for s,r in aux.items():
 assert r['reference']==public[s]['reference']
 for n in ['reference','human']:assert all(type(v) is int and 1<=v<=39 for v in r[n]) and minimum(r[n])<=lengths[cacheix[s]]
assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();print('HEAD_SCREEN_GPU_FREE',free,total,flush=True);assert free>=2*1024**3,(free,total)
torch.cuda.set_per_process_memory_fraction(1*1024**3/total)
heads=nn.ModuleDict()
for name,offsetseed in [('reference',80000),('human',81000)]:
 torch.manual_seed(offsetseed+44);m=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.GELU(),nn.Linear(128,40))
 with torch.no_grad():m[-1].bias.zero_();m[-1].bias[0]=np.log(39)
 heads[name]=m
heads.to('cuda:0');opt=torch.optim.AdamW(heads.parameters(),lr=.001,weight_decay=.001)
initial={k:v.detach().cpu().clone() for k,v in heads.state_dict().items()}
def batch(ids):
 ll=np.asarray([lengths[cacheix[s]] for s in ids]);x=np.zeros((len(ids),int(ll.max()),768),np.float32)
 for j,s in enumerate(ids):
  i=cacheix[s];v=np.asarray(bank[off[i]:off[i]+ll[j]],np.float32);assert np.isfinite(v).all();x[j,:ll[j]]=v
 return torch.from_numpy(x).to('cuda:0'),ll
auxids=sorted(aux);gen=torch.Generator().manual_seed(60044);steps=0;epoch_losses=[]
heads.train()
for epoch in range(3):
 order=torch.randperm(len(auxids),generator=gen).tolist();ls=[]
 for ii in range(0,len(order),32):
  ids=[auxids[i] for i in order[ii:ii+32]];x,ll=batch(ids);loss=0.
  for name in ['reference','human']:
   logits=heads[name](x).to('cpu',dtype=torch.float64).transpose(0,1);lp=logits.log_softmax(-1)
   seq=[aux[s][name] for s in ids];target=torch.tensor([v for row in seq for v in row],dtype=torch.long);tl=torch.tensor([len(r) for r in seq],dtype=torch.long)
   val=F.ctc_loss(lp,target,torch.from_numpy(ll),tl,blank=0,reduction='none',zero_infinity=False)/tl.clamp_min(1)
   loss=loss+.5*val.mean()
  assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
  norm=float(torch.nn.utils.clip_grad_norm_(heads.parameters(),1.,error_if_nonfinite=True));assert norm>0
  opt.step();steps+=1;ls.append(float(loss.detach()))
  if steps%63==0:
   progress(stage='WavLM_own_dual_CTC_heads',epoch=epoch+1,steps=steps,total_steps=945,mean_epoch_loss=float(np.mean(ls)),seconds=time.monotonic()-start)
   print('WAVLM_HEAD_UPDATES',steps,945,float(np.mean(ls)),flush=True)
 epoch_losses.append(float(np.mean(ls)))
assert steps==945
cp=R/'WAVLM_OWN_DUAL_PHONE_SEED44_FINAL_V1.pt'
with cp.open('xb') as f:torch.save(dict(heads=heads.state_dict(),optimizer=opt.state_dict(),initial_heads=initial,seed=44,steps=steps,epochs=3,aux_ids=auxids,aux_ids_sha=digest(auxids),aux_rows=10077,FIT_rows=13340,qualification_sha=sha(Q/'WAVLM_PUBLIC_CACHE_QUALIFICATION_V1.json'),index_sha=p['index_sha'],encoder_optimizer_updates=0),f)
put('WAVLM_CTC_HEAD_TRAINING_RESULT_V1.json',dict(actual_updates=steps,seed=44,aux_rows=len(auxids),all_aux_in_FIT=True,head_parameters=sum(v.numel() for v in heads.parameters()),epoch_losses=epoch_losses,checkpoint_sha=sha(cp),encoder_optimizer_updates=0,peer_supervised_artifacts_used=False))
heads.eval();out=np.empty((len(train),35),np.float64)
with torch.no_grad():
 for i in range(0,len(train),32):
  ids=train[i:i+32];x,ll=batch(ids);lps={name:heads[name](x).double().cpu().log_softmax(-1).numpy() for name in ['reference','human']}
  for j,s in enumerate(ids):out[i+j]=dual35(lps['reference'][j,:ll[j]].copy(),lps['human'][j,:ll[j]].copy(),public[s])
  if (i//32)%32==0:
   progress(stage='WavLM_own_actual35_features',processed=min(i+len(ids),len(train)),total=len(train),seconds=time.monotonic()-start)
   print('WAVLM_COUNT_FEATURES',i+len(ids),len(train),flush=True)
save('WAVLM_OWN_SEED44_TRAIN_FEATURES35_V1.npz',signals=np.asarray(train),features=out)
with np.load(p['qwen_readout'],allow_pickle=False) as z:
 ss=z['FIT_signals'].tolist()+z['held_signals'].tolist();xx=np.r_[z['FIT_features'],z['held_features']];ix={s:i for i,s in enumerate(ss)}
 qx=np.asarray([xx[ix[s]] for s in train]);prior=z['held_prediction'].copy();assert z['held_signals'].tolist()==held
with np.load(p['count_targets'],allow_pickle=False) as z:assert z['signals'].tolist()==train;y=z['y'].copy()
pred={}
for name,x in [('wavlm35',out),('wavlm35_plus_qwen39',np.c_[out,qx])]:
 qq,norm=ridge(x,fi,y);pred[name]=qq[he];save('WAVLM_SCREEN_'+name+'_READOUT_V1.npz',**norm)
save('WAVLM_ALL_HELD_PREDICTIONS_SEALED_V1.npz',signals=np.asarray(held),qwen39=prior,**pred)
ms={k:metric(v,y[he]) for k,v in pred.items()};base=metric(prior,y[he]);winner=min(ms,key=lambda n:ms[n]['RMSE']);gain=base['RMSE']-ms[winner]['RMSE']
put('WAVLM_HEAD_FAST_SCREEN_RESULT_V1.json',dict(status='single_seed_screen_complete',role='TRAINheld2840_not_original4059',FIT_rows=13340,held_rows=2840,seed=44,metrics=ms,baseline_Qwen39=base,selected=winner,gain_points=gain,promising=gain>=.05,head_updates=945,encoder_updates=0,new_count_ridge_fits=2,formal_DEV_gold_read=False,seconds=time.monotonic()-start,GPU_allocated_peak_bytes=torch.cuda.max_memory_allocated(),limits=p['limits']))
print('WAVLM_HEAD_SCREEN',ms,base,winner,gain,flush=True)
