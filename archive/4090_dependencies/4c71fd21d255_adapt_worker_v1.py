import os,sys,pathlib,json,hashlib,time,ast,math,re
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'ADAPT_PLAN_V1.json').read_text());S=pathlib.Path(p['native_screen_root'])
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
from torch import nn
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
 nodes=[n for n in ast.parse(pathlib.Path(f).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f),'exec'),globals())
def selected_responses(text,wanted):
 assert text.lstrip().startswith('{');i=text.index('{')+1;found={}
 while i<len(text):
  while text[i].isspace()or text[i]==',':i+=1
  if text[i]=='}':break
  assert text[i]=='"';a=i;i+=1;escaped=False
  while True:
   c=text[i];i+=1
   if escaped:escaped=False
   elif c=='\\':escaped=True
   elif c=='"':break
  key=json.loads(text[a:i])
  while text[i].isspace():i+=1
  assert text[i]==':';i+=1
  while text[i].isspace():i+=1
  a=i;assert text[i]in '[{';depth=0;quoted=False;escaped=False
  while i<len(text):
   c=text[i];i+=1
   if quoted:
    if escaped:escaped=False
    elif c=='\\':escaped=True
    elif c=='"':quoted=False
   elif c=='"':quoted=True
   elif c in '[{':depth+=1
   elif c in ']}':
    depth-=1
    if depth==0:break
  if key in wanted:
   assert key not in found;value=json.loads(text[a:i]);assert isinstance(value,list)and len(value)==1 and isinstance(value[0].get('response'),str);found[key]=value[0]['response'].strip()
 assert set(found)==wanted
 return found
def partition(logits,selected):
 z=logits.double();lz=torch.logsumexp(z,dim=-1,keepdim=True);mask=torch.ones(z.shape[-1],dtype=torch.bool,device=z.device);mask[selected]=False
 return torch.cat([z[:,selected]-lz,torch.logsumexp(z[:,mask],dim=-1,keepdim=True)-lz],dim=-1)
def ctc_batch(logits,lens,seqs):
 selected=sorted({0}|{t for seq in seqs for t in seq});mapping={v:i for i,v in enumerate(selected)};assert selected[0]==0
 lp=partition(logits,selected).cpu();pieces=torch.split(lp,lens,dim=0);padded=nn.utils.rnn.pad_sequence(pieces,batch_first=False,padding_value=0.)
 targets=torch.tensor([mapping[t]for seq in seqs for t in seq],dtype=torch.long);tl=torch.tensor([len(s)for s in seqs],dtype=torch.long)
 loss=torch.nn.functional.ctc_loss(padded,targets,torch.tensor(lens),tl,blank=0,reduction='none',zero_infinity=False)/tl.clamp_min(1)
 assert torch.isfinite(loss).all();return loss
class Head(nn.Module):
 def __init__(self,seed):
  super().__init__();torch.manual_seed(seed);self.A=nn.Parameter(torch.randn(8,1024)*.02);self.B=nn.Parameter(torch.zeros(16384,8))
 def forward(self,x):return torch.nn.functional.linear(x,W,b)+(x@self.A.T)@self.B.T
start=time.monotonic()
for n,h in read(R/'ADAPT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
for k in ['roles','micro_roles','count_targets','word_public','auxiliary','responses','model_source']:assert sha(p[k])==p[k+'_sha'],k
sp=read(S/'VOCAB_PLAN_V1.json');assert sha(S/'vocab_worker_v1.py')=='e1b3d678888621a26c22124f0c75ae24418c68a2e8c0a412bffd529eaf16da25';defs(S/'vocab_worker_v1.py',['ctc_stats','words','extract_feature'])
source=pathlib.Path(p['protocol_root'])/'wavlm_head_worker_v1.py';assert sha(source)==p['protocol_sha'];defs(source,['ridge','edits'])
with np.load(p['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();FIT=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
assert digest(train)==p['TRAIN_ID_SHA']and digest(dev)==p['DEV_ID_SHA']
aux=read(p['auxiliary']);assert len(aux)==10077 and set(aux)<=set(FIT)
responses=selected_responses(pathlib.Path(p['responses']).read_text(),set(aux));assert all(t.lower()!='xxx'for t in responses.values())
public=read(p['word_public']);M=pathlib.Path(p['snapshot']);assert sha(M/'tokenizer.json')==p['tokenizer_sha'];tok=Tokenizer.from_file(str(M/'tokenizer.json'))
C=pathlib.Path(p['cache']['root']);assert sha(C/'index.json')==p['cache']['index_sha'];index=read(C/'index.json');ci={s:i for i,s in enumerate(index['signal_ids'])}
mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
def frames(s):
 i=ci[s];off=index['offsets'][i];n=index['lengths'][i];return np.asarray(mm[off:off+n],dtype=np.float32).copy()
labels={}
for s in aux:
 r=tok.encode(public[s]['prompt'],add_special_tokens=False).ids;h=tok.encode(responses[s],add_special_tokens=False).ids;n=index['lengths'][ci[s]]
 if all(len(t)+sum(a==b for a,b in zip(t[:-1],t[1:]))<=n for t in [r,h]):
  assert r and all(0<v<16384 for v in r+h);labels[s]=dict(ref=r,response=h)
ids=sorted(labels,key=lambda s:digest(['native-BPE-adapter2048-v1',s]))[:2048];assert len(ids)==2048 and set(ids)<=set(FIT)and not set(ids)&set(held+dev)
put('ADAPT_FIT_BPE_LABELS_BEFORE_TRAINING_V1.json',{s:labels[s]for s in ids});put('ADAPT_SUPERVISION_SCOPE_BEFORE_TRAINING_V1.json',dict(aux_old_accepted=10077,bpe_feasible=len(labels),selected=2048,selected_ID_SHA=digest(ids),human_source_sha=p['responses_sha'],unselected_response_values_json_decoded=False,formal_DEV_or_held_answers=False,source_case_preserved=True,head_rank=8,planned_updates=768))
with np.load(p['micro_roles'],allow_pickle=False)as z:cf=z['FIT'].tolist();ch=z['held'].tolist()
assert len(cf)==4096 and len(ch)==1024 and set(cf)<=set(FIT)and set(ch)<=set(held);ss=cf+ch
save('ADAPT_PUBLIC_ROLES_BEFORE_TRAINING_V1.npz',auxFIT=np.asarray(ids),countFIT=np.asarray(cf),countheld=np.asarray(ch))
assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=3*1024**3;torch.cuda.set_per_process_memory_fraction(2*1024**3/total)
with safe_open(str(M/'model.safetensors'),framework='pt',device='cpu')as f:W=f.get_tensor('encoder.out.weight').float().to('cuda');b=f.get_tensor('encoder.out.bias').float().to('cuda')
assert not W.requires_grad and not b.requires_grad
hs=nn.ModuleDict({name:Head(seed)for name,seed in [('ref',80044),('response',81044)]}).to('cuda');assert sum(v.numel()for v in hs.parameters())==278528
proof=[]
for s in ids[:2]:
 with torch.no_grad():x=torch.from_numpy(frames(s)).to('cuda');warm=torch.nn.functional.linear(x,W,b).double().cpu()
 for name in ['ref','response']:
  seq=labels[s][name];L=max(len(seq),1);n=len(warm);a=warm.clone().requires_grad_();dense=torch.nn.functional.ctc_loss(a.log_softmax(-1)[:,None],torch.tensor(seq,dtype=torch.long),torch.tensor([n]),torch.tensor([len(seq)]),blank=0,reduction='sum')/L;ga,=torch.autograd.grad(dense,a)
  btest=warm.clone().requires_grad_();compact=ctc_batch(btest,[n],[seq]).sum();gb,=torch.autograd.grad(compact,btest)
  lgap=float(abs(dense-compact));ggap=float((ga-gb).abs().max());assert lgap<1e-8 and ggap<1e-8
  proof.append(dict(signal=s,head=name,full_vs_partition_loss_gap=lgap,full_vs_partition_logits_gradient_gap=ggap,full_gradient_norm=float(ga.norm())))
put('ADAPT_DENSE_PARTITION_CTC_LOSS_AND_GRADIENT_QUALIFICATION_V1.json',dict(proof=proof,partition='blank+unique_target+OTHER_rest_vocabulary_mass',other_never_target=True,normalized_probabilities=True,unrenormalized_slice_not_used_for_training=True))
opt=torch.optim.AdamW(hs.parameters(),lr=.001,weight_decay=.001,foreach=False);rng=np.random.default_rng(60044);steps=0
for epoch in range(3):
 order=rng.permutation(len(ids))
 for pos in range(0,len(ids),8):
  batch=[ids[i]for i in order[pos:pos+8]];opt.zero_grad(set_to_none=True);parts=[];cur=[];nt=0
  for s in batch:
   nnf=index['lengths'][ci[s]]
   if cur and nt+nnf>640:parts.append(cur);cur=[];nt=0
   cur.append(s);nt+=nnf
  if cur:parts.append(cur)
  loss_total=0.
  for part in parts:
   lens=[index['lengths'][ci[s]]for s in part];x=torch.from_numpy(np.concatenate([frames(s)for s in part])).to('cuda');assert not x.requires_grad
   for name in ['ref','response']:
    logits=hs[name](x);loss=ctc_batch(logits,lens,[labels[s][name]for s in part]).sum()*(.5/len(batch));assert torch.isfinite(loss);loss.backward();loss_total+=float(loss.detach());del logits,loss
  if steps==0:
   gradients={name:dict(B_norm=float(hs[name].B.grad.norm()),A_norm=float(hs[name].A.grad.norm()))for name in hs};assert all(v['B_norm']>0 for v in gradients.values());put('ADAPT_FIRST_UPDATE_GRADIENTS_V1.json',gradients)
  torch.nn.utils.clip_grad_norm_(hs.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
  if steps%64==0:progress(stage='native_BPE_adapter_training',steps=steps,total=768,loss=loss_total,seconds=time.monotonic()-start)
assert steps==768
cp=R/'ADAPT_DUAL_NATIVE_BPE_HEADS_SEED44_V1.pt'
with cp.open('xb')as f:torch.save(dict(weights=hs.state_dict(),optimizer=opt.state_dict(),steps=steps,seed=44,head_rank=8,base_weight_key='encoder.out.weight',base_head_sha=read(S/'VOCAB_HEAD_PROVENANCE_BEFORE_FEATURES_V1.json')['weight_tensor_sha'],aux_ID_SHA=digest(ids)),f)
all_features={name:[]for name in hs};all_text={name:[]for name in hs}
with torch.inference_mode():
 for k,s in enumerate(ss):
  x=torch.from_numpy(frames(s)).to('cuda')
  for name in ['ref','response']:
   lp=hs[name](x).double().log_softmax(-1).cpu().numpy();f,text,_=extract_feature(s,lp,lp.argmax(1),tok,public);all_features[name].append(f);all_text[name].append(text)
  if(k+1)%128==0:progress(stage='native_BPE_adapter_count_features',processed=k+1,total=len(ss),seconds=time.monotonic()-start)
feat={k:np.asarray(v)for k,v in all_features.items()}
save('ADAPT_NATIVE96_TRAIN_MICRO_FEATURES_V1.npz',signals=np.asarray(ss),ref48=feat['ref'],human48=feat['response'],ref_text=np.asarray(all_text['ref']),human_text=np.asarray(all_text['response']))
with np.load(p['qwen_readout'],allow_pickle=False)as z:
 qs=z['FIT_signals'].tolist()+z['held_signals'].tolist();qx=np.r_[z['FIT_features'],z['held_features']];qi={s:i for i,s in enumerate(qs)};q39=np.asarray([qx[qi[s]]for s in ss])
with np.load(S/'VOCAB_NATIVE48_TRAIN_MICRO_FEATURES_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==ss;frozen=z['features'].copy()
with np.load(p['count_targets'],allow_pickle=False)as z:assert z['signals'].tolist()==train;yt=z['y'].copy()
ti={s:i for i,s in enumerate(train)};y=np.asarray([yt[ti[s]]for s in ss]);fits=np.arange(4096);pred={}
basis={'Q39':q39,'Q39frozen87':np.c_[q39,frozen],'ref48':feat['ref'],'human48':feat['response'],'dual96':np.c_[feat['ref'],feat['response']],'Q39human87':np.c_[q39,feat['response']],'Q39dual135':np.c_[q39,feat['ref'],feat['response']]}
for name,X in basis.items():
 q,nm=ridge(X,fits,y);pred[name]=q[4096:];save('ADAPT_'+name+'_READOUT_V1.npz',**nm)
save('ADAPT_ALL_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(ch),**pred)
metrics={k:metric(q,y[4096:])for k,q in pred.items()};assert abs(metrics['Q39frozen87']['RMSE']-28.14937933299718)<1e-8
eligible=['Q39dual135','Q39human87','dual96','human48','ref48'];selected=min(eligible,key=lambda n:metrics[n]['RMSE']);gain=metrics['Q39frozen87']['RMSE']-metrics[selected]['RMSE']
result=dict(status='native_BPE_adaptation_micro_complete',metrics=metrics,selected=selected,gain=gain,promising=bool(gain>.05),head_updates=steps,aux_rows=2048,head_parameters=278528,new_encoder_updates=0,countFIT=4096,countheld=1024,original_DEV_gold_or_features=False,seconds=time.monotonic()-start,limits=p['limits'])
put('ADAPT_FAST_SCREEN_RESULT_V1.json',result);put('ADAPT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print(json.dumps(result),flush=True)
