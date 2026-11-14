import os,sys,pathlib,json,hashlib,time,ast,re,math
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'WORD_TEXT_PLAN_V1.json').read_text());S=pathlib.Path(p['native_screen_root']);A=pathlib.Path(p['adapt_screen_root'])
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
def alignment(ref,hyp):
 nr=len(ref);nh=len(hyp);d=np.zeros((nr+1,nh+1),dtype=np.int64);d[:,0]=np.arange(nr+1);d[0]=np.arange(nh+1)
 for i in range(1,nr+1):
  for j in range(1,nh+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
 maps=[None]*nr;insert=np.zeros(nr);i=nr;j=nh
 while i or j:
  if i and j and ref[i-1]==hyp[j-1]and d[i,j]==d[i-1,j-1]:maps[i-1]=hyp[j-1];i-=1;j-=1
  elif i and j and d[i,j]==d[i-1,j-1]+1:maps[i-1]=hyp[j-1];i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:i-=1
  else:insert[min(i,nr-1)]+=1;j-=1
 out=[]
 for k,(r,h)in enumerate(zip(ref,maps)):
  distance=sum(edits(list(r),list(h or '')))/max(len(r),len(h or ''),1)
  out.append([float(h==r),float(h is not None and h!=r),float(h is None),distance,min(len(h or '')/max(len(r),1),3.),insert[k]/nr])
 return np.asarray(out),maps
def local_features(ref,texts,severity,contractions):
 arrays=[];maps=[]
 for t in texts:
  v,m=alignment(ref,expanded_words(t,contractions));arrays.append(v);maps.append(m)
 n=len(ref);pub=np.asarray([[math.log1p(len(word)),j/max(n-1,1),1/(j+1),ref.count(word)/n,math.log1p(n),float(severity==0),float(severity==1),float(severity==2),float(j==n-1)]for j,word in enumerate(ref)])
 frozen=np.c_[arrays[0],pub];agreement=np.asarray([[float(maps[a][j]is not None and maps[a][j]==maps[b][j])for a,b in [(0,1),(0,2),(1,2)]]for j in range(n)])
 three=np.c_[arrays[0],arrays[1],arrays[2],agreement,pub];assert frozen.shape==(n,15)and three.shape==(n,30)
 return frozen,three
def build(dim,seed):
 torch.manual_seed(88000+seed);net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1))
 with torch.no_grad():net[-1].weight.zero_();net[-1].bias.zero_()
 return net
def forward(net,x,anchor,legal):
 logits=torch.logit(anchor.clamp(1e-5,1-1e-5))[:,None]+net(x).squeeze(-1);prob=torch.sigmoid(logits);q=(prob*legal).sum(1)/legal.sum(1)
 return q,logits
start=time.monotonic()
for n,h in read(R/'WORD_TEXT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['word_targets','normalization_source','assets_protocol']:assert sha(p[k])==p[k.replace('_source','')+'_sha']if k=='normalization_source'else sha(p[k])==p[k+'_sha']
defs(p['normalization_source'],['expanded_words']);contractions=read(p['assets_protocol'])['contractions'];sp=read(S/'VOCAB_PLAN_V1.json')
source=pathlib.Path(sp['protocol_root'])/'wavlm_head_worker_v1.py';assert sha(source)==sp['protocol_sha'];defs(source,['ridge','edits'])
assert read(A/'ADAPT_DRIVER_TERMINAL_V1.json')['actual_exit_code']==0
with np.load(S/'VOCAB_NATIVE48_TRAIN_MICRO_FEATURES_V1.npz',allow_pickle=False)as z:ss=z['signals'].tolist();native=z['features'].copy();ftext=z['greedy_text'].tolist()
with np.load(A/'ADAPT_NATIVE96_TRAIN_MICRO_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;rtext=z['ref_text'].tolist();htext=z['human_text'].tolist()
with np.load(sp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();FIT=set(z['FIT_ids'].tolist());held=set(z['historical_held_ids'].tolist())
N=4096;assert len(ss)==5120 and set(ss[:N])<=FIT and set(ss[N:])<=held
public=read(sp['word_public']);legal=np.zeros((len(ss),12),dtype=bool);l15=np.zeros((len(ss),12,15));l30=np.zeros((len(ss),12,30))
for i,s in enumerate(ss):
 n=len(public[s]['words']);assert 1<=n<=12;legal[i,:n]=True;a,b=local_features(public[s]['words'],[ftext[i],rtext[i],htext[i]],public[s]['severity'],contractions);l15[i,:n]=a;l30[i,:n]=b
save('WORD_TEXT_LOCAL_FEATURES_BEFORE_TRAINING_V1.npz',signals=np.asarray(ss),frozen15=l15,three30=l30,legal=legal)
with np.load(p['word_targets'],allow_pickle=False)as z:
 assert z['signals'].tolist()==train;ti={s:i for i,s in enumerate(train)};ii=[ti[s]for s in ss[:N]];word_y=z['word_targets'][ii].copy();known=z['word_target_mask'][ii].copy();assert np.all(known<=legal[:N])
with np.load(sp['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;y=np.asarray([z['y'][ti[s]]for s in ss])
with np.load(sp['qwen_readout'],allow_pickle=False)as z:
 qs=z['FIT_signals'].tolist()+z['held_signals'].tolist();qx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qs)};q39=np.asarray([qx[qi[s]]for s in ss])
X=np.c_[q39,native];pred,nm=ridge(X,np.arange(N),y);anchor=pred.copy();fold=np.asarray([int(digest(['native87-nonlinear-oof-v1',' '.join(public[s]['words'])])[:8],16)%5 for s in ss[:N]])
save('WORD_TEXT_ANCHOR_RIDGE_NORMALIZER_V1.npz',**nm);save('WORD_TEXT_TRAIN_FOLDS_V1.npz',signals=np.asarray(ss[:N]),fold=fold)
for f in range(5):
 q,n=ridge(X,np.where(fold!=f)[0],y);ii=np.where(fold==f)[0];anchor[ii]=q[ii];save('WORD_TEXT_ANCHOR_OOF_FOLD%d_V1.npz'%f,**n)
ZG=np.clip((np.where(np.isfinite(X),X,nm['median'])-nm['mean'])/nm['std'],-6,6);AT=torch.tensor(anchor,dtype=torch.float32);LT=torch.tensor(legal,dtype=torch.float32);YT=torch.tensor(y,dtype=torch.float32);WY=torch.tensor(word_y,dtype=torch.float32);KM=torch.tensor(known)
preds=dict(Ridge87=pred[N:]);proof=[];models=[]
for mode,L in [('frozen_only',l15),('three_text',l30)]:
 valid=L[:N][legal[:N]];mu=valid.mean(0);sd=np.maximum(valid.std(0),1e-4);ZL=np.clip((L-mu)/sd,-6,6);XX=np.concatenate([np.repeat(ZG[:,None,:],12,axis=1),ZL],axis=2);XT=torch.tensor(XX,dtype=torch.float32);net=build(XX.shape[-1],44);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(99044);steps=0
 save('WORD_TEXT_'+mode+'_NORM_V1.npz',word_mean=mu,word_std=sd)
 for epoch in range(8):
  order=torch.randperm(N,generator=rng).numpy()
  for pos in range(0,N,128):
   ii=order[pos:pos+128];q,z=forward(net,XT[ii],AT[ii],LT[ii]);loss=(q-YT[ii]).square().mean()
   if KM[ii].any():loss=loss+.1*F.binary_cross_entropy_with_logits(z[KM[ii]],WY[ii][KM[ii]])
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
 assert steps==256;cp='WORD_TEXT_'+mode+'_MODEL_SEED44_V1.pt'
 with(R/cp).open('xb')as f:torch.save(dict(weights=net.state_dict(),optimizer=opt.state_dict(),steps=steps,seed=44,mode=mode,dimension=XX.shape[-1],FIT_ids_sha=digest(ss[:N]),known_word_labels=int(known.sum())),f)
 with torch.no_grad():q,z=forward(net,XT[N:],AT[N:],LT[N:]);q=q.double().numpy()
 W={k:v.numpy().astype(np.float64)for k,v in net.state_dict().items()};hh=XT[N:].double().numpy()@W['0.weight'].T+W['0.bias'];out=((hh*ndtr(hh))@W['2.weight'].T+W['2.bias'])[...,0];aa=AT[N:].double().numpy().clip(1e-5,1-1e-5);pp=expit(np.log(aa/(1-aa))[:,None]+out);oq=(pp*legal[N:]).sum(1)/legal[N:].sum(1);gap=float(np.max(abs(oq-q)));assert gap<2e-6;preds[mode]=q;proof.append(dict(mode=mode,NumPy_all_held_wordmean_maxgap=gap));models.append(dict(mode=mode,steps=steps,checkpoint_sha=sha(R/cp)))
save('WORD_TEXT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(ss[N:]),**preds)
metrics={k:metric(q,y[N:])for k,q in preds.items()};assert abs(metrics['Ridge87']['RMSE']-28.14937933299718)<1e-8;selected=min(['frozen_only','three_text'],key=lambda n:metrics[n]['RMSE']);gain=metrics['Ridge87']['RMSE']-metrics[selected]['RMSE'];result=dict(status='native_BPE_word_identity_fastscreen_complete',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>.03),new_score_updates=512,known_FIT_word_labels=int(known.sum()),independent_NumPy=proof,models=models,formal_DEV_gold_or_features=False,seconds=time.monotonic()-start)
put('WORD_TEXT_FAST_SCREEN_RESULT_V1.json',result);put('WORD_TEXT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
