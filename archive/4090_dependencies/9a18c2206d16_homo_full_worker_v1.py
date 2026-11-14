import os,sys,pathlib,json,hashlib,time,ast,math,re,csv
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'HOMO_FULL_PLAN_V1.json').read_text());S=pathlib.Path(p['screen_root']);H=pathlib.Path(p['word_full_root']);B=pathlib.Path(p['native_full_root']);Q=pathlib.Path(p['soft_word_root']);A=pathlib.Path(p['adapt_screen_root'])
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from functools import lru_cache
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
start=time.monotonic()
for n,h in read(R/'HOMO_FULL_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['soft_source','screen_worker','screen_result','native_full_plan','word_full_plan','adapt_checkpoint','cmu','word_targets']:assert sha(p[k])==p[k+'_sha'],k
assert read(S/'HOMOPHONE_DRIVER_TERMINAL_V1.json')['actual_exit_code']==0 and read(p['screen_result'])['promising']and read(p['screen_result'])['selected']=='combined168'
bp=read(p['native_full_plan']);wp=read(p['word_full_plan']);defs(p['soft_source'],['token_word_map','soft_word_stats']);defs(p['screen_worker'],['sequence_distance','phone_pair','phonetic_alignment'])
defs(pathlib.Path(p['word_text_screen'])/'word_text_worker_v1.py',['alignment','build','forward']);assert sha(p['normalization_source'])==p['normalization_sha'];defs(p['normalization_source'],['expanded_words']);source=pathlib.Path(bp['protocol_root'])/'wavlm_head_worker_v1.py';assert sha(source)==bp['protocol_sha'];defs(source,['edits']);defs(A/'adapt_worker_v1.py',['Head'])
for k in ['roles','count_targets','word_public']:assert sha(bp[k])==bp[k+'_sha']
with np.load(bp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
N=len(train);assert N==16180 and len(dev)==4059 and digest(train)==bp['TRAIN_ID_SHA']and digest(dev)==bp['DEV_ID_SHA'];ss=train+dev;ix={s:i for i,s in enumerate(ss)}
public=read(bp['word_public']);contractions=read(p['assets_protocol'])['contractions']
with np.load(H/'WORD_FULL_ALL20239_LOCAL30_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;hard=z['local'].copy();legal=z['legal'].copy()
with np.load(Q/'SOFT_WORD33_TRAIN_MICRO_FEATURES_V1.npz',allow_pickle=False)as z:micro=z['signals'].tolist();microsoft=z['features'].copy();assert np.array_equal(z['legal'],np.asarray([legal[ix[s]]for s in micro]))
soft=np.zeros((len(ss),12,33));reuse=set(micro);assert len(reuse)==5120 and reuse<=set(train)
for s,f in zip(micro,microsoft):soft[ix[s]]=f
C=pathlib.Path(bp['cache']['root']);assert sha(C/'index.json')==bp['cache']['index_sha'];index=read(C/'index.json');ci={s:i for i,s in enumerate(index['signal_ids'])};mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
def frames(s):
 i=ci[s];off=index['offsets'][i];n=index['lengths'][i];return np.asarray(mm[off:off+n],dtype=np.float32).copy()
M=pathlib.Path(bp['snapshot']);assert sha(M/'tokenizer.json')==bp['tokenizer_sha'];tok=Tokenizer.from_file(str(M/'tokenizer.json'));todo=[s for s in ss if s not in reuse];tokens={s:token_word_map(public[s]['words'],tok)for s in todo}
assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=3*1024**3;torch.cuda.set_per_process_memory_fraction(2*1024**3/total)
with safe_open(str(M/'model.safetensors'),framework='pt',device='cpu')as f:W=f.get_tensor('encoder.out.weight').float().to('cuda');b=f.get_tensor('encoder.out.bias').float().to('cuda')
hs=nn.ModuleDict({name:Head(seed)for name,seed in [('ref',80044),('response',81044)]}).to('cuda');cp=torch.load(p['adapt_checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==768 and cp['seed']==44;hs.load_state_dict(cp['weights'])
groups=[];cur=[];nf=0;maxgap=0.;missing=0;done=0
for s in todo:
 n=index['lengths'][ci[s]]
 if cur and nf+n>1024:groups.append(cur);cur=[];nf=0
 cur.append(s);nf+=n
if cur:groups.append(cur)
with torch.inference_mode():
 for group in groups:
  lens=[index['lengths'][ci[s]]for s in group];x=torch.from_numpy(np.concatenate([frames(s)for s in group])).to('cuda');base=F.linear(x,W,b)
  for h,name in enumerate(['frozen','ref','response']):
   logits=base if name=='frozen'else base+(x@hs[name].A.T)@hs[name].B.T;lz=torch.logsumexp(logits.double(),dim=1);maxprob=torch.exp(logits.max(1).values.double()-lz);pos=0
   for s,n in zip(group,lens):
    ids,mapping=tokens[s];sl=logits[pos:pos+n];zz=lz[pos:pos+n];lp=(sl[:,ids].double()-zz[:,None]).cpu().numpy();bl=(sl[:,0].double()-zz).cpu().numpy();mp=maxprob[pos:pos+n].cpu().numpy();nw=len(public[s]['words']);feat,logZ,gap=soft_word_stats(lp,bl,mp,ids,mapping,nw);soft[ix[s],:nw,11*h:11*(h+1)]=feat;maxgap=max(maxgap,gap);missing+=int(feat[:,-1].sum());pos+=n
  done+=len(group)
  if done%256<len(group):progress(stage='soft_word_remaining15119',processed=done,total=len(todo),seconds=time.monotonic()-start)
del hs,W,b;torch.cuda.empty_cache();save('HOMO_FULL_ALL20239_SOFT33_V1.npz',signals=np.asarray(ss),features=soft,legal=legal)
put('HOMO_FULL_FROZEN_FEATURE_REUSE_AND_SCOPE_V1.json',dict(reused_micro5120=True,new_public_rows=len(todo),all4059_public_only=True,max_new_posterior_mass_gap=maxgap,missing_word_head_cases=missing,no_legal_rows_removed=True,prior_dense_CTC_qualification_sha=sha(Q/'SOFT_CTC_POSTERIOR_QUALIFICATION_V1.json'),prior_packed_head_qualification_sha=sha(H/'WORD_FULL_TEXT_PACK_QUALIFICATION_V1.json'),new_ASR_updates=0,shared_BPE_adapter_seed44=True))
with np.load(H/'WORD_FULL_ALL20239_ADAPTED_GREEDY_TEXTS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;rtext=z['ref_text'].tolist();htext=z['human_text'].tolist()
with np.load(B/'VOCAB_FULL_NATIVE48_ALL20239_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;native=z['features'].copy();ftext=z['greedy_text'].tolist()
dictionary={}
for line in pathlib.Path(p['cmu']).read_text().splitlines():
 a=line.split()
 if len(a)<2 or a[0].startswith(';;;'):continue
 word=re.sub(r'\(\d+\)$','',a[0]).lower();pron=tuple(re.sub(r'\d','',v)for v in a[1:])
 if pron and pron not in dictionary.get(word,[]):dictionary.setdefault(word,[]).append(pron)
phone=np.zeros((len(ss),12,18))
with np.load(S/'HOMOPHONE_NEW_LOCAL_FEATURES_V1.npz',allow_pickle=False)as z:ms=z['signals'].tolist();vv=z['phone18'].copy();assert ms==micro
for s,v in zip(ms,vv):phone[ix[s]]=v
for i,s in enumerate(ss):
 if s in reuse:continue
 ref=public[s]['words'];n=len(ref);ee=[]
 for text in [ftext[i],rtext[i],htext[i]]:
  aa,e,mm_=phonetic_alignment(ref,expanded_words(text,contractions));ee.append(e)
 phone[i,:n]=np.c_[ee[0],ee[1],ee[2]]
 if(i+1)%1024==0:progress(stage='phonetic_word_features',processed=i+1,total=len(ss),seconds=time.monotonic()-start)
local=np.concatenate([hard,phone,soft],axis=2);assert local.shape==(len(ss),12,81)and np.isfinite(local).all();save('HOMO_FULL_ALL20239_LOCAL81_V1.npz',signals=np.asarray(ss),local=local,legal=legal)
valid=local[:N][legal[:N]];mu=valid.mean(0);sd=np.maximum(valid.std(0),1e-4);ZL=np.clip((local-mu)/sd,-6,6);save('HOMO_FULL_WORD_NORM_V1.npz',word_mean=mu,word_std=sd)
with np.load(p['word_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;wy=z['word_targets'].copy();known=z['word_target_mask'].copy();assert np.all(known<=legal[:N])
with np.load(bp['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;y=z['y'].copy()
assert sha(p['parent_prediction'])==p['parent_prediction_sha']and sha(p['old_word_prediction'])==p['old_word_prediction_sha']
with np.load(p['parent_prediction'],allow_pickle=False)as z:assert z['signals'].tolist()==dev;parents=[z[k].copy()for k in p['parent_keys']];oparents=[z['oracle_'+k].copy()for k in p['parent_keys']]
with np.load(p['old_word_prediction'],allow_pickle=False)as z:assert z['signals'].tolist()==dev;oldq=[z[k].copy()for k in p['old_word_keys']];oldoq=[z[k].copy()for k in p['old_word_oracle_keys']]
with np.load(H/'WORD_FULL_TRAIN_FOLDS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==train;fold=z['fold'].copy()
LT=torch.tensor(legal,dtype=torch.float32);YT=torch.tensor(y,dtype=torch.float32);WY=torch.tensor(wy,dtype=torch.float32);KM=torch.tensor(known);pred={'baseline_best_mean':(parents[0]+parents[1])/2};models=[];proof=[]
for j,seed in enumerate([44,45]):
 with np.load(bp['qwen_readouts'][str(seed)],allow_pickle=False)as z:
  qq=z['FIT_signals'].tolist()+z['held_signals'].tolist();xx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qq)};q39=np.asarray([xx[qi[s]]for s in train])
 with np.load(bp['qwen_DEV'][str(seed)],allow_pickle=False)as z:assert z['signals'].tolist()==dev;q39=np.r_[q39,z['features']]
 raw=np.c_[q39,native];rf=B/('VOCAB_FULL_concat87_READOUT_SEED%d_V1.npz'%seed)
 with np.load(rf,allow_pickle=False)as z:nm={k:z[k].copy()for k in z.files}
 ZG=np.clip((np.where(np.isfinite(raw),raw,nm['median'])-nm['mean'])/nm['std'],-6,6);anchor=np.clip(np.c_[np.ones(len(ss)),ZG]@nm['coef'],0,1)
 for f in range(5):
  file=H/('WORD_FULL_ANCHOR_SEED%d_FOLD%d_V1.npz'%(seed,f))
  with np.load(file,allow_pickle=False)as z:st={k:z[k].copy()for k in z.files}
  ii=np.where(fold==f)[0];aa=np.c_[np.ones(len(ii)),np.clip((np.where(np.isfinite(raw[ii]),raw[ii],st['median'])-st['mean'])/st['std'],-6,6)];anchor[ii]=np.clip(aa@st['coef'],0,1)
 XX=np.concatenate([np.repeat(ZG[:,None,:],12,axis=1),ZL],axis=2);XT=torch.tensor(XX,dtype=torch.float32);AT=torch.tensor(anchor,dtype=torch.float32);net=build(168,seed);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(99000+seed);steps=0
 for epoch in range(8):
  order=torch.randperm(N,generator=rng).numpy()
  for pos in range(0,N,128):
   ii=order[pos:pos+128];q,z=forward(net,XT[ii],AT[ii],LT[ii]);loss=(q-YT[ii]).square().mean()
   if KM[ii].any():loss=loss+.1*F.binary_cross_entropy_with_logits(z[KM[ii]],WY[ii][KM[ii]])
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
 assert steps==1016;cpfile='HOMO_FULL_MODEL_SEED%d_V1.pt'%seed
 with(R/cpfile).open('xb')as f:torch.save(dict(weights=net.state_dict(),optimizer=opt.state_dict(),steps=steps,seed=seed,dimension=168,TRAIN_ID_SHA=digest(train),known_word_labels=int(known.sum()),global_norm_source=str(rf),global_norm_sha=sha(rf),shared_BPE_adapter_seed44_sha=p['adapt_checkpoint_sha']),f)
 with torch.no_grad():q,z=forward(net,XT[N:],AT[N:],LT[N:]);q=q.double().numpy()
 state={k:v.numpy().astype(np.float64)for k,v in net.state_dict().items()};hh=XT[N:].double().numpy()@state['0.weight'].T+state['0.bias'];out=((hh*ndtr(hh))@state['2.weight'].T+state['2.bias'])[...,0];aa=AT[N:].double().numpy().clip(1e-5,1-1e-5);pp=expit(np.log(aa/(1-aa))[:,None]+out);oq=(pp*legal[N:]).sum(1)/legal[N:].sum(1);gap=float(np.max(abs(q-oq)));assert gap<2e-6;pred['native%d'%seed]=q;pred['oracle_native%d'%seed]=oq;models.append(dict(seed=seed,file=cpfile,sha=sha(R/cpfile),steps=steps,global_norm_source=str(rf),global_norm_sha=sha(rf)));proof.append(dict(seed=seed,all4059_NumPy_wordmean_maxgap=gap))
 for beta in p['weights']:
  tag='paired'+str(round(beta*100));pred[tag+'_seed%d'%seed]=np.clip(parents[j]+beta*(q-oldq[j]),0,1);pred['oracle_'+tag+'_seed%d'%seed]=np.asarray([min(1.,max(0.,math.fsum([float(a),beta*float(b),-beta*float(c)])))for a,b,c in zip(oparents[j],oq,oldoq[j])])
 progress(stage='combined168_full_score_saved',seed=seed,steps=steps,seconds=time.monotonic()-start)
priority=[]
for beta in p['weights']:
 tag='paired'+str(round(beta*100));priority.append(tag+'_mean');pred[tag+'_mean']=(pred[tag+'_seed44']+pred[tag+'_seed45'])/2;pred['oracle_'+tag+'_mean']=(pred['oracle_'+tag+'_seed44']+pred['oracle_'+tag+'_seed45'])/2
save('HOMO_FULL_ALL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz',signals=np.asarray(dev),**pred)
put('HOMO_FULL_MODELS_AND_PREDICTIONS_SEALED_V1.json',dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),TRAIN_ID_SHA=digest(train),DEV_ID_SHA=digest(dev),TRAIN_rows=N,DEV_rows=4059,models=models,independent_NumPy=proof,source_binding_sha=sha(R/'HOMO_FULL_BINDING_V1.json'),predictions_sha=sha(R/'HOMO_FULL_ALL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz'),local_norm_sha=sha(R/'HOMO_FULL_WORD_NORM_V1.npz'),local_features_sha=sha(R/'HOMO_FULL_ALL20239_LOCAL81_V1.npz'),parent_prediction_sha=p['parent_prediction_sha'],old_word_prediction_sha=p['old_word_prediction_sha'],formal_DEV_gold_read_THIS_run=False,DEV_repeated_development=True,new_score_updates=2032,new_ASR_updates=0,shared_BPE_adapter_seed44=True,priority=priority))
for k in ['truth','canonical_plan','canonical_inputs']:assert sha(bp[k])==bp[k+'_sha']
with pathlib.Path(bp['truth']).open(newline='')as f:rr=list(csv.DictReader(f))
assert [r['signal']for r in rr]==dev;truth=np.asarray([float(r['truth'])for r in rr]);metrics={k:metric(v,truth)for k,v in pred.items()};assert abs(metrics['baseline_best_mean']['RMSE']-p['parent_score'])<1e-8
passing=[k for k in priority if metrics[k]['RMSE']<26 and metrics['oracle_'+k]['RMSE']<26];winner=passing[0]if passing else None
result=dict(status='combined168_homophone_soft_word_original4059_complete',metrics=metrics,goal_strict_less_than26=bool(passing),selected_goal_candidate=winner,actual_score_seeds=[44,45],new_score_updates=2032,shared_BPE_adapter_seed44=True,seconds=time.monotonic()-start,no_extra_CI_gate=True,new_EvalAI_verified=False)
put('HOMO_FULL_ORIGINAL4059_RESULT_V1.json',result);put('HOMO_FULL_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
