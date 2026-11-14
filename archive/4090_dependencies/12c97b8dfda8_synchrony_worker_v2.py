import os,sys,pathlib,json,hashlib,time,ast,math
R=pathlib.Path(__file__).resolve().parent;G=R.parent;H=G.parent
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-a166ebf1-4538-b63c-fe51-571b0fba6a4f',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8');sys.dont_write_bytecode=True
for k in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH']:
 p0=R/'runtime'/k;p0.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p0)
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from scipy.special import logsumexp
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
start=time.monotonic()
def read(f):return json.loads(pathlib.Path(f).read_text())
def nz(f,keys=None):
 with np.load(f,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def put(n,j):
 with(R/n).open('x')as f:json.dump(j,f,indent=2,allow_nan=False)
def save(n,**v):
 with(R/n).open('xb')as f:np.savez_compressed(f,**v)
def note(**v):
 j=dict(seconds=time.monotonic()-start,**v);(R/'PROGRESS_V1.json').write_text(json.dumps(j));print('GRANITE_DISTRIBUTION',json.dumps(j),flush=True)
C=read(G/'PATHS_V1.json');hp=read(pathlib.Path(C['A'])/'HOMO_FULL_PLAN_V1.json');vp=read(hp['native_full_plan']);assets=read(vp['word_public']);roles=nz(vp['roles']);train=roles['strict_TRAIN_ids'].tolist();dev=roles['formal_DEV_ids'].tolist();ss=train+dev;lookup={s:i for i,s in enumerate(ss)};mi=nz(read(G/'PLAN_V1.json')['micro_roles']);fi=mi['FIT'].tolist();held=mi['held'].tolist()if'held'in mi else mi['HELD'].tolist();micro=fi+held;order=[lookup[s]for s in micro];microset=set(micro);other=[j for j,s in enumerate(ss)if s not in microset]
cache=nz(G/'granite_native_CTC_human_LoRA_full_v1/FULL_PUBLIC_GRANITE_FINAL1024_F16_V1.npz');assert cache['signals'].tolist()==ss;hidden=cache['features'];offsets=cache['offsets'];del cache
M=pathlib.Path(vp['snapshot']);assert sha(M/'tokenizer.json')==vp['tokenizer_sha'];tok=Tokenizer.from_file(str(M/'tokenizer.json'))
source=H/'under26_native_BPE_soft_word_posterior_fastscreen_20261020_v1/soft_worker_v1.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='token_word_map');exec(compile(ast.Module(body=[fn],type_ignores=[]),'same_native_word_query','exec'),globals())
free,total=torch.cuda.mem_get_info();assert free>3*1024**3;torch.cuda.set_per_process_memory_fraction(2*1024**3/total)
with safe_open(str(M/'model.safetensors'),framework='pt',device='cpu')as f:W=f.get_tensor('encoder.out.weight').float().to('cuda:0');b=f.get_tensor('encoder.out.bias').float().to('cuda:0')
adaptsource=pathlib.Path(hp['adapt_screen_root'])/'adapt_worker_v1.py';tree=ast.parse(adaptsource.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Head');exec(compile(ast.Module(body=[cl],type_ignores=[]),'existing2048FIT_native_heads','exec'),globals());hs=nn.ModuleDict({n:Head(seed)for n,seed in [('ref',80044),('response',81044)]}).to('cuda:0');assert sha(hp['adapt_checkpoint'])==hp['adapt_checkpoint_sha'];cp=torch.load(hp['adapt_checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==768 and cp['seed']==44;hs.load_state_dict(cp['weights']);hs.eval().requires_grad_(False);W.requires_grad_(False);b.requires_grad_(False)
def occupancy(lp,blank,ids):
 T,L=lp.shape
 if L==0 or L+sum(a==b for a,b in zip(ids[:-1],ids[1:]))>T:return None,None
 ext=np.zeros(2*L+1,np.int64);ext[1::2]=ids;S=len(ext);e=np.empty((T,S));e[:,::2]=blank[:,None];e[:,1::2]=lp;skip=np.zeros(S,bool);skip[2:]=(ext[2:]!=0)&(ext[2:]!=ext[:-2]);a=np.full((T,S),-np.inf);a[0,:2]=e[0,:2]
 for t in range(1,T):v=a[t-1];a[t]=e[t]+np.logaddexp(np.logaddexp(v,np.r_[-np.inf,v[:-1]]),np.where(skip,np.r_[-np.inf,-np.inf,v[:-2]],-np.inf))
 Z=float(np.logaddexp(a[-1,-1],a[-1,-2]));be=np.full((T,S),-np.inf);be[-1,-2:]=0.
 for t in range(T-2,-1,-1):
  v=be[t+1]+e[t+1];be[t]=np.logaddexp(np.logaddexp(v,np.r_[v[1:],-np.inf]),np.where(np.r_[skip[2:],False,False],np.r_[v[2:],-np.inf,-np.inf],-np.inf))
 gamma=np.exp(a+be-Z);assert np.isfinite(gamma).all()and abs(gamma.sum(1)-1).max()<1e-6;return gamma[:,1::2],Z

from synchrony_lib_v1 import extract as synchrony_features
from scipy.special import expit,logit
from scipy.linalg import cho_factor,cho_solve
import csv
tdt=G/'native_TDT_transition_duration_posterior_fast_v5';hist=np.zeros((len(ss),12,3,64),np.float32);legal=np.asarray([[w<len(assets[s]['words'])for w in range(12)]for s in ss]);done=np.zeros(len(ss),bool);proof=[];Y=nz(vp['count_targets'])['y'];nf=len(fi);N=len(train)
@torch.no_grad()
def process(j):
 sig=ss[j];x=torch.from_numpy(hidden[offsets[j]:offsets[j+1]].astype(np.float32)).to('cuda:0');base=F.linear(x,W,b);logits=torch.stack([base,base+(x@hs['ref'].A.T)@hs['ref'].B.T,base+(x@hs['response'].A.T)@hs['response'].B.T]).double();lp=logits.log_softmax(-1);ids,owner=token_word_map(assets[sig]['words'],tok);nw=len(assets[sig]['words'])
 for h in range(3):
  y=lp[h][:,ids].cpu().numpy();blank=lp[h,:,0].cpu().numpy();gamma,Z=occupancy(y,blank,ids)
  if gamma is None:continue
  for w in range(nw):
   jj=np.flatnonzero(owner==w)
   if len(jj):
    pp=gamma[:,jj].sum(1);pp/=pp.sum();hist[j,w,h]=np.diff(np.interp(np.linspace(0,1,65),np.linspace(0,1,len(pp)+1),np.r_[0,np.cumsum(pp)]))
  if len(proof)<6:
   raw=float(F.ctc_loss(lp[h].cpu()[:,None,:],torch.tensor(ids,dtype=torch.long),torch.tensor([len(x)]),torch.tensor([len(ids)]),blank=0,reduction='sum'));gap=abs(raw+Z);mass=hist[j,:nw,h].sum(-1);assert gap<1e-8 and np.max(abs(mass[mass>0]-1))<2e-7;proof.append(dict(signal=sig,head=h,dense_fullV_CTC_gap=gap,mass_gap=float(abs(mass[mass>0]-1).max())))
 done[j]=True
cached=nz(R/'MICRO_GRANITE_WORD_TIME64_V1.npz');seal=read(R/'MICRO_GRANITE_QUALIFIED_V1.json');assert sha(R/'MICRO_GRANITE_WORD_TIME64_V1.npz')==seal['sha'];assert cached['signals'].tolist()==micro;hist[order]=cached['histogram'];done[order]=True;proof=seal['proof']
assert sha(tdt/'MICRO_PUBLIC_WORD_DURATION48_V1.npz')==read(tdt/'MICRO_FEATURE_QUALIFICATION_V1.json')['feature_sha']
para=nz(tdt/'MICRO_PUBLIC_WORD_DURATION48_V1.npz',keys=['signals','alignment64']);assert para['signals'].tolist()==micro;raw=nz(pathlib.Path(C['PREV'])/'PUBLIC_RAW_SUPPORT_AND_BASES_V1.npz',keys=['signals','support']);assert raw['signals'].tolist()==ss;support=raw['support'];sev=np.asarray([assets[s]['severity']for s in ss]);features=synchrony_features(para['alignment64'],hist[order],legal[order],sev[order],support[order]);wrong=synchrony_features(para['alignment64'],hist[order],legal[order],sev[order],support[order],True)['cross72'];features['wrong_word72']=wrong
tree=ast.parse((G/'paired_word_moment_count_refit_v2/fit_pooled_pair_v1.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='fit_count');exec(compile(ast.Module(body=[fn],type_ignores=[]),'same_Count_fit','exec'),globals())
def predict(x,st):return expit(np.c_[np.ones(len(x)),np.clip((np.where(np.isfinite(x),x,st['median'])-st['mean'])/st['std'],-6,6)]@st['coef'])
cc=nz(G/'lowrank_cross_evidence_Count_v1/COUNT_PUBLIC_MATRICES_V1.npz');assert cc['signals'].tolist()==ss and cc['micro_signals'].tolist()==micro;old=nz(pathlib.Path(C['NEW_GOAL_ROOT'])/'joint_sameword_audio2_band_v1/MICRO_band_audio2_152_reg1000_V1.npz');X=cc['microX'];base=predict(X[nf:],old);pred={'baseline':base};choices=[]
for mode,e in features.items():
 xx=np.c_[X,e];st=fit_count(xx[:nf],Y[np.asarray(order[:nf])],np.r_[old['penalty'][1:],np.full(e.shape[1],1000.)]);save('MICRO_'+mode+'_V1.npz',**st);q=predict(xx[nf:],st)
 for gamma in [1.,.5]:
  key=mode+'_g%d'%round(100*gamma);pred[key]=(1-gamma)*base+gamma*q
  if not mode.startswith('wrong'):choices.append(dict(key=key,mode=mode,gamma=gamma,dim=e.shape[1]))
save('HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(held),**pred);put('HELD_SEAL_V1.json',dict(sha=sha(R/'HELD_PREDICTIONS_BEFORE_METRICS_V1.npz'),DEV_gold_read=False))
def metric(q,y):return dict(RMSE=float(100*np.sqrt(np.mean((q-y)**2))),PCC=float(np.corrcoef(q,y)[0,1]))
metrics={k:metric(q,Y[np.asarray(order[nf:])])for k,q in pred.items()};best=min(choices,key=lambda c:metrics[c['key']]['RMSE']);gain=metrics['baseline']['RMSE']-metrics[best['key']]['RMSE'];put('HELD_RESULT_V1.json',dict(metrics=metrics,choice=best,gain=gain,formal_DEV=False));note(phase='micro_complete',metrics=metrics,best=best)
if gain<.005:put('STOP_NEGATIVE_MICRO_V1.json',dict(gain=gain));sys.exit(0)
# Reuse complete TDT publichist when the duration experiment promoted; otherwise continue exact same frozen extraction only.
if(tdt/'FULL_PUBLIC_WORD_DURATION48_V1.npz').exists():
 pp=nz(tdt/'FULL_PUBLIC_WORD_DURATION48_V1.npz',keys=['signals','alignment64']);assert pp['signals'].tolist()==ss;fullpara=pp['alignment64']
else:
 source=(tdt/'duration_worker_v5.py').read_text();prefix=source[:source.index('features=np.zeros((len(ss),12,48))')];(R/'tdt_frozen_continuation_source_v1.py').write_text(prefix);(R/'duration_posterior_v1.py').write_bytes((tdt/'duration_posterior_v1.py').read_bytes());ns=dict(__file__=str(R/'tdt_frozen_continuation_source_v1.py'),__name__='tdt_frozen_auxiliary');exec(compile(prefix,'exact_TDT_frozen_setup','exec'),ns);assert ns['ss']==ss;ns['alignments']=np.zeros((len(ss),12,64));ns['alignments'][order]=para['alignment64']
 with torch.no_grad():
  for pos,j in enumerate(other):
   ns['extract'](ss[j],False)
   if(pos+1)%512==0:note(phase='full_TDT_word_time',done=pos+1,total=len(other))
 fullpara=ns['alignments'];ns.clear();torch.cuda.empty_cache()
save('ALL20239_TDT_WORD_TIME64_V1.npz',signals=np.asarray(ss),histogram=fullpara)
for pos,j in enumerate(other):
 process(j)
 if(pos+1)%512==0:note(phase='full_Granite_word_time',done=int(done.sum()),total=len(ss))
assert done.all();save('ALL20239_GRANITE_WORD_TIME64_V1.npz',signals=np.asarray(ss),histogram=hist,legal=legal);e=synchrony_features(fullpara,hist,legal,sev,support)[best['mode']];save('ALL20239_SELECTED_SYNCHRONY_FEATURES_V1.npz',signals=np.asarray(ss),features=e);gamma=best['gamma'];M=pathlib.Path(C['NEW_GOAL_ROOT'])/'hearing_modulated_sameword_stability_v1';wp=read(pathlib.Path(C['WIN'])/'JOINT_POOL_FULL_PLAN_V1.json');pa=nz(wp['parent_prediction']);assert pa['signals'].tolist()==dev;out={};baseline=[];records=[]
for seed in [44,45]:
 X=cc['fullX%d'%seed];old=nz(M/('hearing1_both146_FULL_SEED%d_V1.npz'%seed));xx=np.c_[X,e];st=fit_count(xx[:N],Y,np.r_[old['penalty'][1:],np.full(e.shape[1],1000.)]);file='FULL_SEED%d_V1.npz'%seed;save(file,**st);b=predict(X[N:],old);q=(1-gamma)*b+gamma*predict(xx[N:],st);pr=pa['paired5_seed%d'%seed];out['seed%d'%seed]=expit(.5*(logit(pr.clip(1e-6,1-1e-6))+logit(q.clip(1e-6,1-1e-6))));baseline.append(expit(.5*(logit(pr.clip(1e-6,1-1e-6))+logit(b.clip(1e-6,1-1e-6)))));records.append(dict(seed=seed,file=file,sha=sha(R/file)))
out['mean']=(out['seed44']+out['seed45'])/2;out['baseline']=(baseline[0]+baseline[1])/2;out['composite20']=.8*out['baseline']+.2*out['mean'];save('FULL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz',signals=np.asarray(dev),**out);put('FULL_MODELS_AND_PREDICTIONS_SEALED_V1.json',dict(models=records,prediction_sha=sha(R/'FULL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz'),selected=best,DEV_gold_read=False))
assert sha(vp['truth'])==vp['truth_sha']
with pathlib.Path(vp['truth']).open(newline='')as f:rs=list(csv.DictReader(f))
assert[r['signal']for r in rs]==dev;y=np.asarray([float(r['truth'])for r in rs]);metrics={k:metric(q,y)for k,q in out.items()};assert abs(metrics['baseline']['RMSE']-25.84654992410905)<1e-8;bestkey=min(['mean','composite20'],key=lambda k:metrics[k]['RMSE']);put('FULL_ORIGINAL4059_RESULT_V1.json',dict(metrics=metrics,best=bestkey,RMSE=metrics[bestkey]['RMSE'],goal_met=metrics[bestkey]['RMSE']<=25.70,EvalAI_verified=False,DEV_repeated_development=True));note(phase='complete',metrics=metrics)
