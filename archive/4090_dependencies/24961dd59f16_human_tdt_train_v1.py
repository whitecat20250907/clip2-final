import os,sys,pathlib,json,hashlib,time,math,ast
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'HUMAN_TDT_PLAN_V1.json').read_text())
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','TORCH_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','NUMBA_CACHE_DIR']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from selected_response_parser_v1 import selected_responses
from tdt_loss_core_v1 import LowRankJoint,streamed_loss_gradient
from tdt_qualification_v1 import qualify
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def save(n,**v):
 with(R/n).open('xb')as f:np.savez_compressed(f,**v)
start=time.monotonic()
for n,h in read(R/'HUMAN_TDT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['source_ASR_ids','responses','micro_roles','native_plan']:assert sha(p[k])==p[k+'_sha']
np_=read(p['native_plan']);NAT=pathlib.Path(p['native_root']);assert sha(NAT/'native_tdt_core_v1.py')==p['native_core_sha'];hp=read(p['source_full_plan']);bp=read(hp['native_full_plan'])
with np.load(bp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();FIT=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
ids=sorted(read(p['source_ASR_ids']));assert len(ids)==2048 and set(ids)<=set(FIT)and not set(ids)&set(held+dev);responses=selected_responses(pathlib.Path(p['responses']).read_text(),set(ids));public=read(bp['word_public']);sys.path.insert(0,str(NAT));from native_tdt_core_v1 import NativeTDT
C=pathlib.Path(np_['cache']['root']);assert sha(C/'index.json')==np_['cache']['index_sha'];index=read(C/'index.json');ci={s:i for i,s in enumerate(index['signal_ids'])};mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
def frames(s):
 i=ci[s];off=index['offsets'][i];n=index['lengths'][i];return np.asarray(mm[off:off+n],dtype=np.float32).copy()
free,total=torch.cuda.mem_get_info();assert free>3*1024**3;torch.cuda.set_per_process_memory_fraction(min(.9,2*1024**3/total));engine=NativeTDT(np_);assert engine.weight_sha==p['native_weights_sha'];W=engine.net['joint'].head.weight.detach();bias=engine.net['joint'].head.bias.detach();labels={};cache={}
with torch.no_grad():
 for i,s in enumerate(ids):
  ref=engine.tok.encode(public[s]['prompt'],add_special_tokens=False).ids;human=engine.tok.encode(responses[s],add_special_tokens=False).ids;assert ref and all(0<=v<8192 and v!=3 for v in ref+human);lab=dict(ref=ref,human=human);labels[s]=lab;row=dict(E=engine.net['encoder_projector'](torch.tensor(frames(s),device='cuda')).cpu())
  for name,target in lab.items():row[name]=engine.net['decoder'](torch.tensor([[8192]+target],device='cuda',dtype=torch.long),cache=None)[0].cpu()
  cache[s]=row
  if(i+1)%512==0:print(json.dumps(dict(phase='frozen_prefix_cache',completed=i+1,total=len(ids),seconds=time.monotonic()-start)),flush=True)
put('HUMAN_TDT_TARGETS_BEFORE_UPDATES_V1.json',labels)
with(R/'HUMAN_TDT_FROZEN_PREFIX_CACHE_V1.pt').open('xb')as f:torch.save(cache,f)
put('HUMAN_TDT_SUPERVISION_SCOPE_V1.json',dict(FIT_ASR_rows=2048,ID_SHA=digest(ids),human_source_sha=p['responses_sha'],source_ASR_ids_sha=p['source_ASR_ids_sha'],formal_DEV_or_historical_held_human_answers=False,unselected_response_values_JSON_decoded=False,empty_human=sum(not labels[s]['human']for s in ids),source_native_weights_sha=engine.weight_sha,no_encoder_updates=True))
s=ids[0];proof=qualify((W.cpu(),bias.cpu(),cache[s]['E'],cache[s]['ref'],labels[s]['ref']));put('HUMAN_TDT_SEQUENCE_LOSS_GRADIENT_QUALIFICATION_V1.json',dict(proof=proof,ending_nonzero_blank=True,full8193_normalization=True,sigma=0,omega=0))
hs=nn.ModuleDict({name:LowRankJoint(W,bias,seed)for name,seed in [('ref',80044),('human',81044)]}).to('cuda');assert sum(v.numel()for v in hs.parameters())==141408
with torch.no_grad():
 x=torch.relu(cache[s]['E'][:2].to('cuda')+cache[s]['ref'][0:1].to('cuda'));base=F.linear(x,W,bias)
 for name in hs:assert torch.equal(hs[name](x),base)
opt=torch.optim.AdamW(hs.parameters(),lr=.0005,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(60044);steps=0;loss_history=[]
for epoch in range(3):
 order=torch.randperm(len(ids),generator=rng).tolist()
 for pos in range(0,len(ids),8):
  batch=[ids[i]for i in order[pos:pos+8]];opt.zero_grad(set_to_none=True);value=0.
  for name in ['ref','human']:
   for sid in batch:
    row=cache[sid];E=row['E'].to('cuda');D=row[name].to('cuda');loss=streamed_loss_gradient(hs[name],E,D,labels[sid][name],scale=.5/len(batch));value+=.5*loss/len(batch)
  assert math.isfinite(value);norm=float(torch.nn.utils.clip_grad_norm_(hs.parameters(),1.,error_if_nonfinite=True));opt.step();steps+=1;loss_history.append(value)
  if steps%32==0:print(json.dumps(dict(phase='TDT_LoRA_train',steps=steps,total=768,loss=value,gradient_norm_before_clip=norm,seconds=time.monotonic()-start)),flush=True)
 file='HUMAN_TDT_LORA_STEP%d_V1.pt'%steps
 with(R/file).open('xb')as f:torch.save(dict(heads={name:dict(A=hs[name].A.detach().cpu(),B=hs[name].B.detach().cpu())for name in hs},optimizer=opt.state_dict(),steps=steps,epoch=epoch+1,source_native_weights_sha=engine.weight_sha,ASR_ID_SHA=digest(ids),rank=8,alpha=16),f)
assert steps==768 and all(int(v['step'])==768 for v in opt.state_dict()['state'].values());h=hashlib.sha256()
for k,v in sorted(engine.net.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
assert h.hexdigest()==engine.weight_sha and all(not v.requires_grad and v.grad is None for v in engine.net.parameters())
for name in hs:assert torch.equal(hs[name].W,W)and torch.equal(hs[name].b,bias)and float(hs[name].B.norm())>0
put('HUMAN_TDT_ADAPTERS_SEALED_V1.json',dict(actual_optimizer_steps=768,ASR_ID_SHA=digest(ids),checkpoint='HUMAN_TDT_LORA_STEP768_V1.pt',checkpoint_sha=sha(R/'HUMAN_TDT_LORA_STEP768_V1.pt'),prefix_cache_sha=sha(R/'HUMAN_TDT_FROZEN_PREFIX_CACHE_V1.pt'),frozen_base_tensor_sha_after_training=h.hexdigest(),loss_first32=float(np.mean(loss_history[:32])),loss_last32=float(np.mean(loss_history[-32:])),head_roles=['ref','human'],shared_adapter_seed44=True,new_encoder_updates=0))
with np.load(p['micro_roles'],allow_pickle=False)as z:cf=z['FIT'].tolist();ch=z['held'].tolist()
ss=cf+ch;assert len(cf)==4096 and len(ch)==1024 and set(cf)<=set(FIT)and set(ch)<=set(held)
@torch.no_grad()
def independent_decode(array,head,cap):
 E=engine.net['encoder_projector'](torch.tensor(array,device='cuda'));history=[8192];frame=0;tokens=[];durations=[];positions=[];values=[]
 for step in range(cap):
  if frame>=len(array):break
  emb=F.embedding(torch.tensor([history],device='cuda'),engine.net['decoder'].embedding.weight);h,_=engine.net['decoder'].lstm(emb);D=F.linear(h[:,-1],engine.net['decoder'].decoder_projector.weight,engine.net['decoder'].decoder_projector.bias);joint=F.relu(E[frame:frame+1]+D);z=(F.linear(joint,head.W,head.b)+2*F.linear(F.linear(joint,head.A),head.B))[0];token=int(z[:8193].argmax());dur=int(z[8193:].argmax());dur=1 if token==8192 and dur==0 else dur
  tokens.append(token);durations.append(dur);positions.append(frame);values.append(z.cpu().numpy());frame+=dur
  if token==3:break
  if token!=8192:history.append(token)
 return dict(raw_tokens=tokens,durations=durations,frames=positions),values
newheadproof=[]
for name in ['ref','human']:
 engine.net['joint'].head=hs[name];records={}
 with torch.no_grad():
  arrays=[frames(s)for s in cf[:2]];native,traces=engine.decode_many(arrays,trace_first=2);cap=engine.cfg.max_symbols_per_step*max(len(a)for a in arrays)-1
  for j,array in enumerate(arrays):
   ref,raw=independent_decode(array,hs[name],cap);assert all(native[j][k]==ref[k]for k in ['raw_tokens','durations','frames']);gap=max(float(np.max(abs(a-b)))for a,b in zip(traces[j],raw));assert gap<2e-4;newheadproof.append(dict(head=name,signal=cf[j],full_history_vs_cache_logit_maxgap=gap,paths_exact=True))

 with torch.no_grad():
  for pos in range(0,len(ss),32):
   rows=ss[pos:pos+32];values,_=engine.decode_many([frames(s)for s in rows])
   records.update(zip(rows,values))
   if(pos+len(rows))%512==0:print(json.dumps(dict(phase='adapted_greedy',head=name,completed=pos+len(rows),total=len(ss),seconds=time.monotonic()-start)),flush=True)
 put('HUMAN_TDT_'+name.upper()+'_MICRO_RECORDS_V1.json',records)
put('HUMAN_TDT_ADAPTED_GREEDY_QUALIFICATION_V1.json',dict(tests=newheadproof))
put('HUMAN_TDT_ASR_TRAIN_TERMINAL_V1.json',dict(actual_exit_code=0,optimizer_steps=steps,seconds=time.monotonic()-start));print(json.dumps(dict(phase='ASR_adapter_and_micro_decode_done',steps=steps,seconds=time.monotonic()-start)),flush=True)
