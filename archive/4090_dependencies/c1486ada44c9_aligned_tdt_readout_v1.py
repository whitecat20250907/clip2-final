import os,sys,pathlib,json,hashlib,time
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'ALIGNED_TDT_PLAN_V1.json').read_text());H=pathlib.Path(p['homophone_root'])
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
start=time.monotonic()
for n,h in read(R/'ALIGNED_TDT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
assert read(R/'ALIGNED_TDT_EXTRACT_TERMINAL_V1.json')['actual_exit_code']==0
hp=read(H/'HOMOPHONE_PLAN_V1.json');S=pathlib.Path(hp['native_screen_root']);sp=read(S/'VOCAB_PLAN_V1.json')
with np.load(R/'ALIGNED_TDT_MICRO_LOCAL32_V1.npz',allow_pickle=False)as z:ss=z['signals'].tolist();extra=z['features32'].copy()
with np.load(p['micro_roles'],allow_pickle=False)as z:assert ss==z['FIT'].tolist()+z['held'].tolist()
N=4096
with np.load(S/'VOCAB_NATIVE48_TRAIN_MICRO_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;native=z['features'].copy()
with np.load(H/'HOMOPHONE_LOCAL_FEATURES_BEFORE_TRAINING_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;local30=z['three30'].copy();legal=z['legal'].copy()
with np.load(H/'HOMOPHONE_NEW_LOCAL_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;phone18=z['phone18'].copy()
with np.load(pathlib.Path(hp['soft_word_root'])/'SOFT_WORD33_TRAIN_MICRO_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;soft33=z['features'].copy()
local81=np.concatenate([local30,phone18,soft33],-1)
with np.load(H/'HOMOPHONE_combined168_NORM_V1.npz',allow_pickle=False)as z:ZL=np.clip((local81-z['word_mean'])/z['word_std'],-6,6)
with np.load(H/'HOMOPHONE_ANCHOR_RIDGE_NORMALIZER_V1.npz',allow_pickle=False)as z:nm={k:z[k].copy()for k in z.files}
with np.load(sp['qwen_readout'],allow_pickle=False)as z:
 ids=z['FIT_signals'].tolist()+z['held_signals'].tolist();qx=np.r_[z['FIT_features'],z['held_features']];ii={s:i for i,s in enumerate(ids)};q39=np.asarray([qx[ii[s]]for s in ss])
raw=np.c_[q39,native];ZG=np.clip((np.where(np.isfinite(raw),raw,nm['median'])-nm['mean'])/nm['std'],-6,6);anchor=np.clip(np.c_[np.ones(len(ss)),ZG]@nm['coef'],0,1)
with np.load(H/'HOMOPHONE_TRAIN_FOLDS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss[:N];fold=z['fold'].copy()
for f in range(5):
 with np.load(H/('HOMOPHONE_ANCHOR_OOF_FOLD%d_V1.npz'%f),allow_pickle=False)as z:st={k:z[k].copy()for k in z.files}
 jj=np.where(fold==f)[0];a=np.c_[np.ones(len(jj)),np.clip((np.where(np.isfinite(raw[jj]),raw[jj],st['median'])-st['mean'])/st['std'],-6,6)];anchor[jj]=np.clip(a@st['coef'],0,1)
v=extra[:N][legal[:N]];mu=v.mean(0);sd=np.maximum(v.std(0),1e-4);save('ALIGNED_TDT_FIT32_NORM_V1.npz',mean=mu,std=sd);XX=np.concatenate([np.repeat(ZG[:,None,:],12,axis=1),ZL,np.clip((extra-mu)/sd,-6,6)],-1);XT=torch.tensor(XX,dtype=torch.float32);AT=torch.tensor(anchor,dtype=torch.float32);LT=torch.tensor(legal,dtype=torch.float32)
with np.load(sp['count_targets'],allow_pickle=False)as z:train=z['signals'].tolist();ti={s:i for i,s in enumerate(train)};y=np.asarray([z['y'][ti[s]]for s in ss])
with np.load(hp['word_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;ii=[ti[s]for s in ss[:N]];WY=torch.tensor(z['word_targets'][ii],dtype=torch.float32);KM=torch.tensor(z['word_target_mask'][ii])
YT=torch.tensor(y,dtype=torch.float32);cp=torch.load(H/'HOMOPHONE_combined168_MODEL_SEED44_V1.pt',map_location='cpu',weights_only=False);assert cp['steps']==256
def warm():
 net=nn.Sequential(nn.Linear(168,32),nn.GELU(),nn.Linear(32,1));net.load_state_dict(cp['weights']);return net
def predict(net,dim):
 z=net(XT[N:,:,:dim]).squeeze(-1)+torch.logit(AT[N:].clamp(1e-5,1-1e-5))[:,None];return ((z.sigmoid()*LT[N:]).sum(1)/LT[N:].sum(1)).double().numpy()
with torch.no_grad():base=predict(warm(),168)
with np.load(H/'HOMOPHONE_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss[N:];assert np.max(abs(base-z['combined168']))<1e-8
pred={'baseline168':base};models=[];proof=[]
for mode,dim,epochs,lr in [('fresh200',200,8,.001),('warm200',200,2,.0003),('continue168',168,2,.0003)]:
 torch.manual_seed(88044);net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1))
 with torch.no_grad():
  if mode=='fresh200':net[-1].weight.zero_();net[-1].bias.zero_()
  else:
   old=warm();net[0].weight.zero_();net[0].weight[:,:168].copy_(old[0].weight);net[0].bias.copy_(old[0].bias);net[2].weight.copy_(old[2].weight);net[2].bias.copy_(old[2].bias);assert np.max(abs(predict(net,dim)-base))<2e-7
 opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(99044 if mode=='fresh200'else 99144);steps=0
 for epoch in range(epochs):
  order=torch.randperm(N,generator=rng).numpy()
  for pos in range(0,N,128):
   jj=order[pos:pos+128];z=net(XT[jj,:,:dim]).squeeze(-1)+torch.logit(AT[jj].clamp(1e-5,1-1e-5))[:,None];q=(z.sigmoid()*LT[jj]).sum(1)/LT[jj].sum(1);loss=(q-YT[jj]).square().mean()
   if KM[jj].any():loss=loss+.1*F.binary_cross_entropy_with_logits(z[KM[jj]],WY[jj][KM[jj]])
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
 assert steps==epochs*32;file='ALIGNED_TDT_'+mode+'_MODEL_SEED44_V1.pt'
 with(R/file).open('xb')as f:torch.save(dict(weights=net.state_dict(),optimizer=opt.state_dict(),steps=steps,dimension=dim,mode=mode,prior_steps=0 if mode=='fresh200'else 256,FIT_ID_SHA=digest(ss[:N])),f)
 with torch.no_grad():q=predict(net,dim)
 st={k:v.numpy().astype(np.float64)for k,v in net.state_dict().items()};X=XT[N:,:,:dim].double().numpy();h=X@st['0.weight'].T+st['0.bias'];z=((h*ndtr(h))@st['2.weight'].T+st['2.bias'])[...,0];a=AT[N:].clamp(1e-5,1-1e-5).double().numpy();z+=np.log(a/(1-a))[:,None];oq=(expit(z)*legal[N:]).sum(1)/legal[N:].sum(1);gap=float(np.max(abs(q-oq)));assert gap<2e-6;pred[mode]=q;pred[mode+'_half']=.5*q+.5*base;models.append(dict(mode=mode,steps=steps,checkpoint_sha=sha(R/file)));proof.append(dict(mode=mode,all1024_NumPy_wordmean_maxgap=gap))
save('ALIGNED_TDT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(ss[N:]),**pred);metrics={k:metric(v,y[N:])for k,v in pred.items()};priority=['fresh200','warm200','continue168','fresh200_half','warm200_half','continue168_half'];selected=min(priority,key=lambda k:metrics[k]['RMSE']);gain=metrics['baseline168']['RMSE']-metrics[selected]['RMSE'];result=dict(status='aligned_native_TDT_fullposterior_micro_complete',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>0),models=models,independent_NumPy=proof,new_score_updates=384,new_ASR_updates=0,original_DEV_selected=False,seconds=time.monotonic()-start)
put('ALIGNED_TDT_SCREEN_RESULT_V1.json',result);put('ALIGNED_TDT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
