import os,sys,pathlib,json,hashlib,time,ast,re,math
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'VOCAB_PLAN_V1.json').read_text())
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
from scipy.special import logsumexp
from safetensors import safe_open
from tokenizers import Tokenizer
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v):
 with (R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def save(n,**v):
 with (R/n).open('xb')as f:np.savez_compressed(f,**v)
def progress(**v):
 f=R/'progress.next.json';f.write_text(json.dumps(v));os.replace(f,R/'progress.json')
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
def ctc_stats(lp,tokens,blanklp):
 L=len(tokens);T=len(lp);ext=np.zeros(2*L+1,dtype=np.int64);ext[1::2]=tokens
 if not L or L+sum(a==b for a,b in zip(tokens[:-1],tokens[1:]))>T:return np.nan,np.nan,[np.nan]*4
 uniq=sorted(set(tokens));mapping={v:i for i,v in enumerate(uniq)}
 v=np.column_stack([blanklp]+[lp[:,mapping[t]]for t in tokens])
 cols=np.zeros(len(ext),dtype=np.int64);cols[1::2]=np.arange(1,L+1)
 e=v[:,cols];S=len(ext);skip=np.zeros(S,dtype=bool);skip[2:]=(ext[2:]!=0)&(ext[2:]!=ext[:-2])
 a=np.full(S,-np.inf);a[0]=e[0,0];a[1]=e[0,1];best=a.copy();back=np.zeros((T,S),np.int16)
 for t in range(1,T):
  prev=np.r_[-np.inf,a[:-1]];two=np.r_[-np.inf,-np.inf,a[:-2]];two=np.where(skip,two,-np.inf)
  a=np.logaddexp(np.logaddexp(a,prev),two)+e[t]
  ways=np.stack([best,np.r_[-np.inf,best[:-1]],np.where(skip,np.r_[-np.inf,-np.inf,best[:-2]],-np.inf)])
  ix=ways.argmax(0);back[t]=ix;best=ways[ix,np.arange(S)]+e[t]
 end=S-1 if best[-1]>=best[-2]else S-2;viterbi=float(best[end]);path=np.empty(T,np.int16);path[-1]=end
 for t in range(T-1,0,-1):path[t-1]=path[t]-back[t,path[t]]
 support=[]
 for j in range(L):
  vals=e[path==2*j+1,2*j+1];support.append(float(np.exp(vals).mean())if len(vals)else 0.)
 support=np.asarray(support)
 return -float(np.logaddexp(a[-1],a[-2]))/L,viterbi/L,[float(support.mean()),float(support.min()),float(support.std()),float(np.quantile(support,.25))]
def words(text):return re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text.lower())
def extract_feature(s,logp,ids,tok,public):
 T=len(logp);P=np.exp(logp);ent=-(P*logp).sum(1)/math.log(16384);mx=P.max(1);blank=P[:,0];raw=ids.tolist();seq=[v for i,v in enumerate(raw)if v!=0 and(i==0 or v!=raw[i-1])]
 text=tok.decode(seq,skip_special_tokens=True);hyp=words(text);ref=public[s];nw=max(len(ref['words']),1)
 feat=[];proof=[]
 for j,target in enumerate([ref['prompt'],ref['original_prompt']]):
  target_ids=tok.encode(target,add_special_tokens=False).ids;assert all(0<v<16384 for v in target_ids)
  u=sorted(set(target_ids));reduced=logp[:,u];nll,vit,sp=ctc_stats(reduced,target_ids,logp[:,0]);L=max(len(target_ids),1);rr=words(target);sub,de,ins=edits(rr,hyp);den=max(len(rr),1)
  maxs=np.exp(reduced.max(0))if u else np.array([np.nan])
  row=[nll,nll*L/T,float(not np.isfinite(nll)),sub/den,de/den,ins/den,1-(sub+de+ins)/den,vit]+sp+[float(maxs.mean()),float(maxs.min()),float(np.log(maxs.clip(1e-30)).mean()),L/T,sum(a==b for a,b in zip(target_ids[:-1],target_ids[1:]))/L,float(len(target_ids))]
  assert len(row)==18;feat.extend(row)
  proof.append(dict(tokens=target_ids,nll=nll))
 feat.extend([float(ent.mean()),float(ent.std()),float(blank.mean()),float(blank.std()),float(mx.mean()),float(mx.std()),len(seq)/T,len(hyp)/nw,math.log1p(T),nw,float(ref['severity']==1),float(ref['severity']==2)])
 assert len(feat)==48
 return np.asarray(feat),text,proof
start=time.monotonic()
for n,h in read(R/'VOCAB_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['roles','micro_roles','count_targets','word_public','qwen_readout','model_source']:assert sha(p[k])==p[k+'_sha'],k
S=pathlib.Path(p['protocol_root'])/'wavlm_head_worker_v1.py';assert sha(S)==p['protocol_sha'];nodes=[n for n in ast.parse(S.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in ['ridge','edits']];assert len(nodes)==2;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(S),'exec'))
with np.load(p['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();fit=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
assert len(train)==16180 and digest(train)==p['TRAIN_ID_SHA']and digest(dev)==p['DEV_ID_SHA']
with np.load(p['micro_roles'],allow_pickle=False)as z:fi=z['FIT'].tolist();he=z['held'].tolist()
assert len(fi)==4096 and len(he)==1024 and set(fi)<=set(fit)and set(he)<=set(held)
ss=fi+he;save('VOCAB_PUBLIC_ROLES_BEFORE_FEATURES_V1.npz',FIT=np.asarray(fi),held=np.asarray(he))
C=pathlib.Path(p['cache']['root']);assert sha(C/'index.json')==p['cache']['index_sha']and sha(p['cache']['source'])==p['cache']['source_sha']
index=read(C/'index.json');assert index['dim']==1024 and index['dtype']=='float16'and index['n']==20297
ci={s:i for i,s in enumerate(index['signal_ids'])};assert set(ss)<=set(train)
mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
M=pathlib.Path(p['snapshot']);assert sha(M/'tokenizer.json')==p['tokenizer_sha'];tok=Tokenizer.from_file(str(M/'tokenizer.json'))
assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=2*1024**3;torch.cuda.set_per_process_memory_fraction(1024**3/total)
with safe_open(str(M/'model.safetensors'),framework='pt',device='cpu')as f:
 W=f.get_tensor('encoder.out.weight').float().to('cuda');b=f.get_tensor('encoder.out.bias').float().to('cuda')
assert tuple(W.shape)==(16384,1024)and tuple(b.shape)==(16384,)
put('VOCAB_HEAD_PROVENANCE_BEFORE_FEATURES_V1.json',dict(weight_key='encoder.out.weight',bias_key='encoder.out.bias',tie_mapping='ctc_head.weight/bias -> encoder.out.weight/bias',cached_input='final_encoder_block_output,no_postnorm,source_bound',weight_tensor_sha=hashlib.sha256(W.cpu().numpy().tobytes()+b.cpu().numpy().tobytes()).hexdigest(),vocab=16384,blank=0,interpolation=False,encoder_or_head_updates=0,formal_DEV_features_or_gold=False))
public=read(p['word_public']);rows=[];texts=[];proofs=[]
with torch.inference_mode():
 for k,s in enumerate(ss):
  i=ci[s];off=index['offsets'][i];n=index['lengths'][i];xx=np.asarray(mm[off:off+n],dtype=np.float32).copy()
  outs=[]
  for off2 in range(0,n,256):
   x=torch.from_numpy(xx[off2:off2+256]).to('cuda');outs.append(torch.nn.functional.linear(x,W,b).double().log_softmax(-1).cpu().numpy())
  lp=np.concatenate(outs);assert lp.shape==(n,16384)and np.isfinite(lp).all()
  f,text,qual=extract_feature(s,lp,lp.argmax(1),tok,public);rows.append(f);texts.append(text)
  if k<2:
   for v in qual:
    tokens=v['tokens'];t=torch.tensor(tokens,dtype=torch.long);actual=float(torch.nn.functional.ctc_loss(torch.from_numpy(lp)[:,None,:],t,torch.tensor([n]),torch.tensor([len(tokens)]),blank=0,reduction='sum')/len(tokens))
    gap=abs(actual-v['nll']);assert gap<1e-8 or(not np.isfinite(actual)and not np.isfinite(v['nll']));proofs.append(dict(signal=s,dense_torch_nll=actual,numpy_compressed_nll=v['nll'],gap=gap))
  if(k+1)%128==0:progress(stage='native_vocabulary_evidence',processed=k+1,total=len(ss),seconds=time.monotonic()-start)
X=np.asarray(rows);save('VOCAB_NATIVE48_TRAIN_MICRO_FEATURES_V1.npz',signals=np.asarray(ss),features=X,greedy_text=np.asarray(texts));put('VOCAB_NUMERIC_QUALIFICATION_V1.json',dict(cases=proofs,dense_vs_compressed_forward_only=True,no_training_gradient_claim=True))
with np.load(p['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;yt=z['y'].copy()
ix={s:i for i,s in enumerate(train)};y=np.asarray([yt[ix[s]]for s in ss])
with np.load(p['qwen_readout'],allow_pickle=False)as z:
 qids=z['FIT_signals'].tolist()+z['held_signals'].tolist();qraw=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qids)};q39=np.asarray([qraw[qi[s]]for s in ss])
fitrows=np.arange(4096);pred={};mods=[]
for name,A in [('Q39',q39),('native48',X),('concat87',np.c_[q39,X])]:
 q,nm=ridge(A,fitrows,y);save('VOCAB_'+name+'_READOUT_V1.npz',**nm);pred[name]=q[4096:];mods.append(dict(name=name,file='VOCAB_'+name+'_READOUT_V1.npz'))
pred['halfmean']=.5*(pred['Q39']+pred['native48'])
save('VOCAB_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(he),**pred)
metrics={name:metric(q,y[4096:])for name,q in pred.items()};priority=['concat87','native48','halfmean'];selected=min(priority,key=lambda n:metrics[n]['RMSE']);gain=metrics['Q39']['RMSE']-metrics[selected]['RMSE']
result=dict(status='native_vocabulary_fastscreen_complete',role='TRAIN4096_held1024_NOT_original4059',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>.05),new_encoder_or_head_updates=0,new_count_ridge_fits=3,seconds=time.monotonic()-start,limits=p['limits'])
put('VOCAB_FAST_SCREEN_RESULT_V1.json',result);put('VOCAB_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
