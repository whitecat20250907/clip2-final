import os,sys,pathlib,json,hashlib,time,ast,zipfile,struct
R=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/lisiyi/clip2_lmy/CLIP2_lmy/my_experiments/d6e_qwen_internalization_autoresearch_20261015/goal_DEV_under25_75_word_local_temporal_20261021_v1')
plan=json.loads((ROOT/'PLAN_AND_GROUP_DATAFLOW_REVIEW_BEFORE_IMPLEMENTATION_V1.json').read_text())
G=pathlib.Path(plan['old_goal_root']); A=pathlib.Path(plan['auditory_root'])
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.dont_write_bytecode=True
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH']:
 p=R/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import numpy as np
from scipy.special import expit,logit
from scipy.linalg import cho_factor,cho_solve
def read(f):return json.loads(pathlib.Path(f).read_text())
def nz(f,keys=None):
 with np.load(f,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def sha(f):
 h=hashlib.sha256()
 with pathlib.Path(f).open('rb')as fd:
  for b in iter(lambda:fd.read(4194304),b''):h.update(b)
 return h.hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,allow_nan=False)
def save(n,**v):
 with(R/n).open('xb')as f:np.savez(f,**v)
def note(**v):print('LOCAL_TEMPORAL',json.dumps(v),flush=True)
def defs(f,names):
 nodes=[n for n in ast.parse(pathlib.Path(f).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f),'exec'),globals());return {n:globals()[n]for n in names}
hp=read(read(G/'PLAN_V1.json')['source_full_plan']);bp=read(hp['native_full_plan'])
for k in ['canonical_inputs','canonical_plan','roles']:assert sha(bp[k])==bp[k+'_sha']
roles=nz(bp['roles']);train=roles['strict_TRAIN_ids'].tolist();dev=roles['formal_DEV_ids'].tolist();ss=train+dev;idx={s:i for i,s in enumerate(ss)}
assert len(train)==16180 and len(dev)==4059
for k,ids in [('TRAIN',train),('DEV',dev)]:assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==bp[k+'_ID_SHA']
mi=nz(read(G/'PLAN_V1.json')['micro_roles']); mids=mi['FIT'].tolist()+mi['held'].tolist();nf=len(mi['FIT']);assert nf==4096 and len(mids)==5120 and set(mids)<=set(train)
assets=read(bp['word_public']); public=read(read(G/'PLAN_V1.json')['public_records'])
def frame_map():
 f=G/'granite_native_CTC_human_LoRA_full_v1/FULL_PUBLIC_GRANITE_FINAL1024_F16_V1.npz';sm=nz(f,keys=['signals','offsets']);assert sm['signals'].tolist()==ss
 with zipfile.ZipFile(f)as z:zi=z.getinfo('features.npy');assert zi.compress_type==zipfile.ZIP_STORED
 with f.open('rb')as fd:
  fd.seek(zi.header_offset);hh=fd.read(30);nl,el=struct.unpack('<HH',hh[26:30]);fd.seek(zi.header_offset+30+nl+el);version=np.lib.format.read_magic(fd);shape,fortran,dtype=np.lib.format._read_array_header(fd,version);offset=fd.tell()
 assert shape[1]==1024 and not fortran
 return np.memmap(f,mode='r',dtype=dtype,shape=shape,offset=offset),sm['offsets'],f
def current_micro_base():
 m=nz(A/'current_basis_raw_phonetic_word_interaction_micro_v1/MATCHED_BASE_AND_NEW_MICRO_FEATURES_BEFORE_TARGETS_V1.npz',keys=['signals','base1360'])
 w=nz(G/'granite_native_CTC_human_LoRA_pilot_v1/MICRO_NATIVE_human_FEATURE48_V1.npz');old=nz(G.parent/'under26_granite_native_vocabulary48_fastscreen_20261020_v1/VOCAB_NATIVE48_TRAIN_MICRO_FEATURES_V1.npz')
 assert m['signals'].tolist()==w['signals'].tolist()==old['signals'].tolist()==mids
 delta=np.subtract(w['features'],old['features'],out=np.zeros_like(w['features']),where=np.isfinite(w['features'])&np.isfinite(old['features']))
 x=np.c_[m['base1360'],delta];assert x.shape==(5120,1408)
 st=nz(A/'Granite_physical_stereo_native_v1/micro_physical_Count_v1/COUNT_mono1408_V1.npz');pen=st['penalty'][1:].copy();pen[-48:]*=.5
 return x,pen
def count_pred(x,st):return expit(np.c_[np.ones(len(x)),np.clip((np.where(np.isfinite(x),x,st['median'])-st['mean'])/st['std'],-6,6)]@st['coef'])
