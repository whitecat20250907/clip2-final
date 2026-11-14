import os,sys,pathlib,json,hashlib,time,ast,math,re
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'SOFT_PLAN_V1.json').read_text());S=pathlib.Path(p['word_text_screen']);A=pathlib.Path(p['adapt_root'])
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr,expit
from safetensors import safe_open
from tokenizers import Tokenizer
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def save(n,**v):
 with(R/n).open('xb')as f:np.savez_compressed(f,**v)
def progress(**v):
 f=R/'progress.next.json';f.write_text(json.dumps(v));os.replace(f,R/'progress.json')
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
def defs(f,names):
 nodes=[n for n in ast.parse(pathlib.Path(f).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f),'exec'),globals())
def token_word_map(words,tok):
 text=' '.join(words);enc=tok.encode(text,add_special_tokens=False);spans=[];j=0
 for w in words:spans.append((j,j+len(w)));j+=len(w)+1
 out=[]
 for a,b in enc.offsets:
  overlap=[max(0,min(b,d)-max(a,c))for c,d in spans]
  if max(overlap)>0:k=int(np.argmax(overlap))
  else:k=next((i for i,(c,d)in enumerate(spans)if c>=b),len(words)-1)
  out.append(k)
 return enc.ids,np.asarray(out,dtype=np.int64)
def soft_word_stats(targetlp,blanklp,maxp,ids,mapping,nword):
 T,L=targetlp.shape;out=np.zeros((nword,11));out[:,-1]=1
 if L==0 or L+sum(a==b for a,b in zip(ids[:-1],ids[1:]))>T:return out,None,0.
 ext=np.zeros(2*L+1,dtype=np.int64);ext[1::2]=ids;Sx=len(ext);e=np.empty((T,Sx));e[:,::2]=blanklp[:,None];e[:,1::2]=targetlp
 skip=np.zeros(Sx,dtype=bool);skip[2:]=(ext[2:]!=0)&(ext[2:]!=ext[:-2]);alpha=np.full((T,Sx),-np.inf);alpha[0,:2]=e[0,:2]
 for t in range(1,T):
  a=alpha[t-1];alpha[t]=e[t]+np.logaddexp(np.logaddexp(a,np.r_[-np.inf,a[:-1]]),np.where(skip,np.r_[-np.inf,-np.inf,a[:-2]],-np.inf))
 logZ=float(np.logaddexp(alpha[-1,-1],alpha[-1,-2]));beta=np.full((T,Sx),-np.inf);beta[-1,-2:]=0.
 for t in range(T-2,-1,-1):
  a=beta[t+1]+e[t+1];one=np.r_[a[1:],-np.inf];two=np.r_[a[2:],-np.inf,-np.inf];allowed=np.r_[skip[2:],False,False];beta[t]=np.logaddexp(np.logaddexp(a,one),np.where(allowed,two,-np.inf))
 gamma=np.exp(alpha+beta-logZ);gap=float(np.max(abs(gamma.sum(1)-1.)));assert gap<1e-6 and np.isfinite(gamma).all();tokenmass=gamma[:,1::2];timegrid=np.arange(T,dtype=np.float64)
 for j in range(nword):
  kk=np.where(mapping==j)[0]
  if not len(kk):continue
  g=tokenmass[:,kk];mass=g.sum(0);assert (mass>0).all();lp=targetlp[:,kk];weighted=(g*lp).sum(0)/mass;peaks=lp.max(0);wg=g.sum(1);total=wg.sum();tm=(timegrid*wg).sum()/total;std=math.sqrt(max(0.,float(((timegrid-tm)**2*wg).sum()/total)))/T
  normg=g/mass;entropy=-(normg*np.log(normg.clip(1e-300))).sum(0)/max(math.log(T),1.)
  out[j]=[weighted.mean(),weighted.min(),weighted.std(),peaks.mean(),peaks.min(),float((wg*np.exp(blanklp)).sum()/total),float((wg*maxp).sum()/total),float(total/T),std,float(entropy.mean()),0.]
 assert np.isfinite(out).all();return out,logZ,gap
start=time.monotonic()
for n,h in read(R/'SOFT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['word_text_worker','adapt_checkpoint','warm_score_checkpoint']:assert sha(p[k])==p[k+'_sha']
assert read(S/'WORD_TEXT_DRIVER_TERMINAL_V1.json')['actual_exit_code']==0
wp=read(S/'WORD_TEXT_PLAN_V1.json');vp=read(pathlib.Path(p['native_screen'])/'VOCAB_PLAN_V1.json');ap=read(A/'ADAPT_PLAN_V1.json');defs(p['word_text_worker'],['forward']);defs(A/'adapt_worker_v1.py',['Head'])
with np.load(S/'WORD_TEXT_LOCAL_FEATURES_BEFORE_TRAINING_V1.npz',allow_pickle=False)as z:ss=z['signals'].tolist();hard=z['three30'].copy();legal=z['legal'].copy()
assert len(ss)==5120;N=4096;public=read(vp['word_public'])
with np.load(vp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();FIT=set(z['FIT_ids'].tolist());held=set(z['historical_held_ids'].tolist())
assert set(ss[:N])<=FIT and set(ss[N:])<=held
C=pathlib.Path(vp['cache']['root']);assert sha(C/'index.json')==vp['cache']['index_sha'];index=read(C/'index.json');ci={s:i for i,s in enumerate(index['signal_ids'])};mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
def frames(s):
 i=ci[s];off=index['offsets'][i];n=index['lengths'][i];return np.asarray(mm[off:off+n],dtype=np.float32).copy()
M=pathlib.Path(vp['snapshot']);assert sha(M/'tokenizer.json')==vp['tokenizer_sha'];tok=Tokenizer.from_file(str(M/'tokenizer.json'));tokens={s:token_word_map(public[s]['words'],tok)for s in ss}
assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=3*1024**3;torch.cuda.set_per_process_memory_fraction(2*1024**3/total)
with safe_open(str(M/'model.safetensors'),framework='pt',device='cpu')as f:W=f.get_tensor('encoder.out.weight').float().to('cuda');b=f.get_tensor('encoder.out.bias').float().to('cuda')
hs=nn.ModuleDict({name:Head(seed)for name,seed in [('ref',80044),('response',81044)]}).to('cuda');cp=torch.load(p['adapt_checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==768;hs.load_state_dict(cp['weights'])
features=np.zeros((len(ss),12,33));proof=[];max_posterior_gap=0.;missing=0;groups=[];cur=[];nf=0
for s in ss:
 n=index['lengths'][ci[s]]
 if cur and nf+n>1024:groups.append(cur);cur=[];nf=0
 cur.append(s);nf+=n
if cur:groups.append(cur)
ix={s:i for i,s in enumerate(ss)};done=0
with torch.inference_mode():
 for group in groups:
  lens=[index['lengths'][ci[s]]for s in group];x=torch.from_numpy(np.concatenate([frames(s)for s in group])).to('cuda');base=F.linear(x,W,b)
  for h,name in enumerate(['frozen','ref','response']):
   logits=base if name=='frozen'else base+(x@hs[name].A.T)@hs[name].B.T
   lz=torch.logsumexp(logits.double(),dim=1);maxprob=torch.exp(logits.max(1).values.double()-lz);pos=0
   for s,n in zip(group,lens):
    ids,mapping=tokens[s];sl=logits[pos:pos+n];z=lz[pos:pos+n];lp=(sl[:,ids].double()-z[:,None]).cpu().numpy();bp=(sl[:,0].double()-z).cpu().numpy();mp=maxprob[pos:pos+n].cpu().numpy();nw=len(public[s]['words']);feat,logZ,gap=soft_word_stats(lp,bp,mp,ids,mapping,nw);features[ix[s],:nw,11*h:11*(h+1)]=feat;max_posterior_gap=max(max_posterior_gap,gap);missing+=int(feat[:,-1].sum())
    if ix[s]<2 and logZ is not None:
     dense=sl.double().log_softmax(-1).cpu();ctc=float(F.ctc_loss(dense[:,None,:],torch.tensor(ids,dtype=torch.long),torch.tensor([n]),torch.tensor([len(ids)]),blank=0,reduction='sum'));dgap=abs(ctc+logZ);assert dgap<1e-8;proof.append(dict(signal=s,head=name,dense_Torch_vs_NumPy_logZ_gap=dgap,posterior_frame_mass_gap=gap))
    pos+=n
  done+=len(group)
  if done%256<len(group):progress(stage='native_word_soft_posterior_features',processed=done,total=len(ss),seconds=time.monotonic()-start)
del hs,W,b;torch.cuda.empty_cache();save('SOFT_WORD33_TRAIN_MICRO_FEATURES_V1.npz',signals=np.asarray(ss),features=features,legal=legal);put('SOFT_CTC_POSTERIOR_QUALIFICATION_V1.json',dict(proof=proof,max_all_frame_mass_gap=max_posterior_gap,missing_word_head_cases=missing,no_legal_rows_removed=True))
with np.load(S/'WORD_TEXT_ANCHOR_RIDGE_NORMALIZER_V1.npz',allow_pickle=False)as z:nm={k:z[k].copy()for k in z.files}
with np.load(pathlib.Path(p['native_screen'])/'VOCAB_NATIVE48_TRAIN_MICRO_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;native=z['features'].copy()
with np.load(vp['qwen_readout'],allow_pickle=False)as z:
 qs=z['FIT_signals'].tolist()+z['held_signals'].tolist();qx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qs)};q39=np.asarray([qx[qi[s]]for s in ss])
with np.load(vp['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;ti={s:i for i,s in enumerate(train)};y=np.asarray([z['y'][ti[s]]for s in ss])
with np.load(wp['word_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;ii=[ti[s]for s in ss[:N]];wy=z['word_targets'][ii].copy();known=z['word_target_mask'][ii].copy()
raw=np.c_[q39,native];ZG=np.clip((np.where(np.isfinite(raw),raw,nm['median'])-nm['mean'])/nm['std'],-6,6);anchor=np.clip(np.c_[np.ones(len(ss)),ZG]@nm['coef'],0,1)
with np.load(S/'WORD_TEXT_TRAIN_FOLDS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss[:N];fold=z['fold'].copy()
for f in range(5):
 with np.load(S/('WORD_TEXT_ANCHOR_OOF_FOLD%d_V1.npz'%f),allow_pickle=False)as z:st={k:z[k].copy()for k in z.files}
 ii=np.where(fold==f)[0];aa=np.c_[np.ones(len(ii)),np.clip((np.where(np.isfinite(raw[ii]),raw[ii],st['median'])-st['mean'])/st['std'],-6,6)];anchor[ii]=np.clip(aa@st['coef'],0,1)
with np.load(S/'WORD_TEXT_three_text_NORM_V1.npz',allow_pickle=False)as z:hmu=z['word_mean'].copy();hsd=z['word_std'].copy()
ZH=np.clip((hard-hmu)/hsd,-6,6);sv=features[:N][legal[:N]];smu=sv.mean(0);ssd=np.maximum(sv.std(0),1e-4);ZS=np.clip((features-smu)/ssd,-6,6);save('SOFT_NEW_WORD_NORM_V1.npz',mean=smu,std=ssd)
old=torch.load(p['warm_score_checkpoint'],map_location='cpu',weights_only=False);assert old['steps']==256;oldW=old['weights'];LT=torch.tensor(legal,dtype=torch.float32);AT=torch.tensor(anchor,dtype=torch.float32);YT=torch.tensor(y,dtype=torch.float32);WY=torch.tensor(wy,dtype=torch.float32);KM=torch.tensor(known);baseX=np.concatenate([np.repeat(ZG[:,None,:],12,axis=1),ZH],axis=2)
with np.load(S/'WORD_TEXT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss[N:];warm_saved=z['three_text'].copy()
preds={'warm_word117':warm_saved};models=[];numeric=[];warmproof=[]
for mode,XX in [('soft150',np.concatenate([baseX,ZS],axis=2)),('continue117',baseX)]:
 XT=torch.tensor(XX,dtype=torch.float32);torch.manual_seed(88044);net=nn.Sequential(nn.Linear(XX.shape[-1],32),nn.GELU(),nn.Linear(32,1))
 with torch.no_grad():
  net[0].weight.zero_();net[0].weight[:,:117].copy_(oldW['0.weight']);net[0].bias.copy_(oldW['0.bias']);net[2].weight.copy_(oldW['2.weight']);net[2].bias.copy_(oldW['2.bias']);q,_=forward(net,XT[N:],AT[N:],LT[N:]);warmgap=float(np.max(abs(q.double().numpy()-warm_saved)))
 assert warmgap<2e-6;warmproof.append(dict(mode=mode,maxgap=warmgap))
 opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(99144);steps=0
 for epoch in range(4):
  order=torch.randperm(N,generator=rng).numpy()
  for pos in range(0,N,128):
   ii=order[pos:pos+128];q,z=forward(net,XT[ii],AT[ii],LT[ii]);loss=(q-YT[ii]).square().mean()
   if KM[ii].any():loss=loss+.1*F.binary_cross_entropy_with_logits(z[KM[ii]],WY[ii][KM[ii]])
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
 assert steps==128;file='SOFT_'+mode+'_MODEL_SEED44_V1.pt'
 with(R/file).open('xb')as f:torch.save(dict(weights=net.state_dict(),optimizer=opt.state_dict(),steps=steps,seed=44,warm_checkpoint_sha=p['warm_score_checkpoint_sha'],warm_steps=256,mode=mode),f)
 with torch.no_grad():q,z=forward(net,XT[N:],AT[N:],LT[N:]);q=q.double().numpy()
 state={k:v.numpy().astype(np.float64)for k,v in net.state_dict().items()};hh=XT[N:].double().numpy()@state['0.weight'].T+state['0.bias'];out=((hh*ndtr(hh))@state['2.weight'].T+state['2.bias'])[...,0];aa=AT[N:].double().numpy().clip(1e-5,1-1e-5);pp=expit(np.log(aa/(1-aa))[:,None]+out);oq=(pp*legal[N:]).sum(1)/legal[N:].sum(1);gap=float(np.max(abs(q-oq)));assert gap<2e-6;preds[mode]=q;numeric.append(dict(mode=mode,NumPy_held_prediction_maxgap=gap));models.append(dict(mode=mode,steps=steps,checkpoint_sha=sha(R/file)))
save('SOFT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(ss[N:]),**preds);metrics={k:metric(v,y[N:])for k,v in preds.items()};assert abs(metrics['warm_word117']['RMSE']-28.067556140609824)<1e-8;selected=min(['soft150','continue117'],key=lambda n:metrics[n]['RMSE']);gain=metrics['warm_word117']['RMSE']-metrics[selected]['RMSE'];result=dict(status='soft_word_posterior_fastscreen_complete',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>.03),soft_vs_continue=metrics['soft150']['RMSE']-metrics['continue117']['RMSE'],models=models,independent_NumPy=numeric,warm_parity=warmproof,new_score_updates=256,new_ASR_updates=0,formal_DEV_gold_or_features=False,seconds=time.monotonic()-start)
put('SOFT_FAST_SCREEN_RESULT_V1.json',result);put('SOFT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
