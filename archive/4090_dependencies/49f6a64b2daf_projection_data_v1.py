import os,sys,pathlib,json,hashlib,time,math
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'PROJ_TDT_PLAN_V1.json').read_text());S=pathlib.Path(p['small_root'])
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','TORCH_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','NUMBA_CACHE_DIR']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
sys.path.insert(0,str(S))
from selected_response_parser_v1 import selected_responses
from tdt_loss_core_v1 import LowRankJoint
sys.path.insert(0,str(R))
from projection_core_v1 import ProjectionPair,AcousticWrapper,ContextWrapper,bridge,qualify_bridge
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
start=time.monotonic()
for n,h in read(R/'PROJ_TDT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['source_ASR_ids','responses','micro_roles','native_plan','small_checkpoint','small_binding','asr_checkpoint','asr_binding']:assert sha(p[k])==p[k+'_sha']
for n,h in read(p['small_binding'])['files'].items():assert sha(S/n)==h
assert read(S/'HUMAN_TDT_SEQUENCE_LOSS_GRADIENT_QUALIFICATION_V1.json')['proof'];hp=read(p['source_full_plan']);bp=read(hp['native_full_plan']);np_=read(p['native_plan']);NAT=pathlib.Path(p['native_root'])
with np.load(bp['roles'],allow_pickle=False)as z:FIT=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
ids=sorted(read(p['source_ASR_ids']));assert len(ids)==10077 and set(ids)<=set(FIT)and not set(ids)&set(held+dev);responses=selected_responses(pathlib.Path(p['responses']).read_text(),set(ids));public=read(bp['word_public']);sys.path.insert(0,str(NAT));from native_tdt_core_v1 import NativeTDT
assert sha(NAT/'native_tdt_core_v1.py')==p['native_core_sha'];C=pathlib.Path(np_['cache']['root']);assert sha(C/'index.json')==np_['cache']['index_sha'];index=read(C/'index.json');ci={s:i for i,s in enumerate(index['signal_ids'])};mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
def frames(s):
 i=ci[s];off=index['offsets'][i];n=index['lengths'][i];return np.asarray(mm[off:off+n],dtype=np.float32).copy()
free,total=torch.cuda.mem_get_info();assert free>3*1024**3;torch.cuda.set_per_process_memory_fraction(min(.9,2*1024**3/total));engine=NativeTDT(np_);assert engine.weight_sha==p['native_weights_sha'];W=engine.net['joint'].head.weight.detach();bias=engine.net['joint'].head.bias.detach();warm=torch.load(p['asr_checkpoint'],map_location='cpu',weights_only=False);assert warm['steps']==1260 and warm['source_native_weights_sha']==engine.weight_sha;labels={};cache={};severities={k:0 for k in [0,1,2]}
with torch.no_grad():
 for i,s in enumerate(ids):
  target=engine.tok.encode(responses[s],add_special_tokens=False).ids;assert all(0<=v<8192 and v!=3 for v in target);sev=public[s]['severity'];assert sev in [0,1,2];severities[sev]+=1;labels[s]=target;cache[s]=dict(E=engine.net['encoder_projector'](torch.tensor(frames(s),device='cuda')).cpu(),D=engine.net['decoder'](torch.tensor([[8192]+target],device='cuda',dtype=torch.long),cache=None)[0].cpu(),severity=sev)
  if(i+1)%1024==0:print(json.dumps(dict(phase='expanded_human_prefix_cache',completed=i+1,total=len(ids),seconds=time.monotonic()-start)),flush=True)
put('PROJ_TDT_HUMAN_TARGETS_BEFORE_UPDATES_V1.json',labels);put('PROJ_TDT_SUPERVISION_SCOPE_V1.json',dict(ASR_rows=10077,ID_SHA=digest(ids),severity_counts=severities,human_source_sha=p['responses_sha'],formal_DEV_or_historical_held_human_answers=False,unselected_response_values_JSON_decoded=False,empty_human=sum(not v for v in labels.values()),prefix_cache_in_memory_only=True))
