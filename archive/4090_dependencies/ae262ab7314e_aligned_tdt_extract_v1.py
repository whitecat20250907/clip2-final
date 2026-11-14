import os,sys,pathlib,json,hashlib,time
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'ALIGNED_TDT_PLAN_V1.json').read_text());S=pathlib.Path(p['small_root'])
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','TORCH_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','NUMBA_CACHE_DIR']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
start=time.monotonic()
for n,h in read(R/'ALIGNED_TDT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['asr_checkpoint','asr_binding','small_binding','micro_roles','native_plan']:assert sha(p[k])==p[k+'_sha']
for n,h in read(p['asr_binding'])['files'].items():assert sha(pathlib.Path(p['route_root'])/n)==h
for n,h in read(p['small_binding'])['files'].items():assert sha(S/n)==h
assert sha(pathlib.Path(p['teacher_root'])/'teacher_support_core_v1.py')==p['teacher_helper_sha'];NAT=pathlib.Path(p['native_root']);sys.path.insert(0,str(NAT));sys.path.insert(0,str(S));sys.path.insert(0,p['teacher_root'])
from native_tdt_core_v1 import NativeTDT
from tdt_loss_core_v1 import LowRankJoint
from aligned_posterior_core_v1 import extract32
hp=read(p['source_full_plan']);bp=read(hp['native_full_plan']);public=read(bp['word_public']);np_=read(p['native_plan'])
with np.load(p['micro_roles'],allow_pickle=False)as z:fit=z['FIT'].tolist();held=z['held'].tolist()
with np.load(bp['roles'],allow_pickle=False)as z:FIT=set(z['FIT_ids'].tolist());HOLD=set(z['historical_held_ids'].tolist());dev=set(z['formal_DEV_ids'].tolist())
ss=fit+held;assert len(fit)==4096 and len(held)==1024 and set(fit)<=FIT and set(held)<=HOLD and not set(ss)&dev
C=pathlib.Path(np_['cache']['root']);assert sha(C/'index.json')==np_['cache']['index_sha'];index=read(C/'index.json');ci={s:i for i,s in enumerate(index['signal_ids'])};mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
free,total=torch.cuda.mem_get_info();assert free>3*1024**3;torch.cuda.set_per_process_memory_fraction(min(.9,2*1024**3/total));engine=NativeTDT(np_);assert engine.weight_sha==p['native_weights_sha'];cp=torch.load(p['asr_checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==1260;heads={}
for sev in [0,1,2]:
 name=str(sev);head=LowRankJoint(engine.net['joint'].head.weight,engine.net['joint'].head.bias,84044+sev).to('cuda')
 with torch.no_grad():head.A.copy_(cp['heads'][name]['A']);head.B.copy_(cp['heads'][name]['B'])
 head.requires_grad_(False);heads[sev]=head
features=np.zeros((5120,12,32));proof=[];maxmass=0.
with torch.no_grad():
 for j,s in enumerate(ss):
  i=ci[s];off=index['offsets'][i];n=index['lengths'][i];array=np.asarray(mm[off:off+n],dtype=np.float32).copy();words=public[s]['words'];head=heads[public[s]['severity']];value,mass=extract32(engine,head,array,words);features[j,:len(words)]=value;maxmass=max(maxmass,mass)
  if j<2:
   independent,imass=extract32(engine,head,array,words,dense=True);gap=float(np.max(abs(value-independent)));assert gap<2e-4;proof.append(dict(signal=s,stream_GPU_vs_dense_CPU_FP64_word32_maxgap=gap,occupancy_label_mass_gap=mass))
  if(j+1)%256==0:print(json.dumps(dict(phase='aligned_full_posterior_word32',completed=j+1,total=5120,seconds=time.monotonic()-start)),flush=True)
with(R/'ALIGNED_TDT_MICRO_LOCAL32_V1.npz').open('xb')as f:np.savez_compressed(f,signals=np.asarray(ss),features32=features)
put('ALIGNED_TDT_FEATURE_QUALIFICATION_V1.json',dict(tests=proof,all5120_label_occupancy_sum_maxgap=maxmass,sourceASR1260steps=True,new_ASR_updates=0,no_DEV_selected=True,features_sha=sha(R/'ALIGNED_TDT_MICRO_LOCAL32_V1.npz')));put('ALIGNED_TDT_EXTRACT_TERMINAL_V1.json',dict(actual_exit_code=0,seconds=time.monotonic()-start))
