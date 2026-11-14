"""Pinned original native feature helpers, plus fixed packing; no labels."""
import pathlib,re,copy,hashlib
import numpy as np
import librosa
import torch
from torch.nn import functional as F
LEGACY_SOURCE_SHA256="0808c2378fb8931358aa7f2bdbc3a520df54e73eac6a8483c1f4a8c3c93a5dac"
RAW45_SOURCE_SHA256='ea5ae397958a525338390a897a1abe30d40dd0473d1f877a9fbdd7e2fdc30e41'
def configure(model,proc,chat_prefix,device,hasher):
 global native,processor,prefix,D,sha
 native=model;processor=proc;prefix=chat_prefix;D=device;sha=hasher
def grid(text,n_audio):
  expanded=processor.replace_multimodal_special_tokens([prefix+text],iter([int(n_audio)]))[0]
  ep=processor.replace_multimodal_special_tokens([prefix],iter([int(n_audio)]))[0]
  encoded=processor.tokenizer(expanded,add_special_tokens=False,return_offsets_mapping=True)
  pre=processor.tokenizer(ep,add_special_tokens=False)['input_ids']
  assert expanded==ep+text and encoded['input_ids'][:len(pre)]==pre
  spans=[m.span() for m in re.finditer(r'\S+',text)];owner=[]
  for left,right in encoded['offset_mapping'][len(pre):]:
   left-=len(ep);right-=len(ep);assert 0<=left<right<=len(text)
   nonblank=[k for k in range(left,right) if not text[k].isspace()]
   if nonblank:
    matches=[j for j,(a,b) in enumerate(spans) if a<=nonblank[0] and nonblank[-1]<b];assert len(matches)==1
    owner.append(matches[0])
   else:
    following=[j for j,(a,b) in enumerate(spans) if a>=right];assert following;owner.append(following[0])
  ids=np.asarray(encoded['input_ids'][len(pre):],np.int64);own=np.asarray(owner,np.int64)
  assert len(ids)==len(own) and set(own)==set(range(len(spans))) and (np.diff(own)>=0).all()
  return dict(ids=ids,owner=own,prefix_length=len(pre),full_ids=np.asarray(encoded['input_ids']),word_count=len(spans))

def prepare(row):
  p=pathlib.Path(row['audio']);stat=p.stat();h=sha(p);wave,_=librosa.load(p,sr=16000,mono=True);now=p.stat();assert (stat.st_size,stat.st_mtime_ns)==(now.st_size,now.st_mtime_ns)
  x=processor(text=[prefix+row['text']],audio=[wave],return_tensors='pt',padding=True,truncation=False).to('cpu').to(torch.bfloat16)
  m=grid(row['text'],int((x['input_ids']==native.config.thinker_config.audio_token_id).sum()))
  assert m['word_count']==row['words'] and np.array_equal(x['input_ids'][0].numpy(),m['full_ids'])
  return x,m,h

def base_hash():
  h=hashlib.sha256()
  for name,param in sorted([(n.replace('.base_layer.','.'),p) for n,p in native.named_parameters() if '.lora_' not in n],key=lambda t:t[0]):
   h.update(name.encode()+b'\0');h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()

def distribution_logits(x,m,null):
  batch={k:v.to(D) for k,v in x.items()};hook=None
  if null:
   def remove_audio(module,args,out):
    result=copy.copy(out);result.last_hidden_state=torch.zeros_like(out.last_hidden_state);return result
   hook=native.thinker.audio_tower.register_forward_hook(remove_audio)
  try:
   with torch.inference_mode():out=native.thinker(**batch,use_cache=False,return_dict=True)
  finally:
   if hook is not None:hook.remove()
  pos=torch.arange(m['prefix_length']-1,batch['input_ids'].shape[1]-1,device=D)
  logits=out.logits[0,pos].float();assert logits.shape==(len(m['ids']),151936) and torch.isfinite(logits).all()
  target=torch.as_tensor(m['ids'],device=D);lp=logits.log_softmax(-1).gather(1,target[:,None]).squeeze(1).cpu().numpy()
  return logits,lp

def gpu_stats(r,n):
  lr=r.double().log_softmax(-1);ln=n.double().log_softmax(-1);pr=lr.exp();pn=ln.exp();lm=torch.logaddexp(lr,ln)-__import__('math').log(2)
  tr=lr.topk(2,dim=-1).values;tn=ln.topk(2,dim=-1).values
  return torch.stack([-(pr*lr).sum(-1),-(pn*ln).sum(-1),tr[:,0]-tr[:,1],tn[:,0]-tn[:,1],(pr*(lr-ln)).sum(-1),(pn*(ln-lr)).sum(-1),.5*((pr*(lr-lm)).sum(-1)+(pn*(ln-lm)).sum(-1)),.5*(pr-pn).abs().sum(-1)],-1).cpu().numpy()

def cpu_stats(r,n):
  r=r.astype(np.float64);n=n.astype(np.float64)
  def normalized(a):
   v=a-a.max(-1,keepdims=True);return v-np.log(np.exp(v).sum(-1,keepdims=True))
  lr=normalized(r);ln=normalized(n);pr=np.exp(lr);pn=np.exp(ln);lm=np.logaddexp(lr,ln)-np.log(2.)
  tr=np.partition(r,-2,axis=-1)[:,-2:];tn=np.partition(n,-2,axis=-1)[:,-2:]
  return np.stack([-(pr*lr).sum(-1),-(pn*ln).sum(-1),tr[:,1]-tr[:,0],tn[:,1]-tn[:,0],(pr*(lr-ln)).sum(-1),(pn*(ln-lr)).sum(-1),.5*((pr*(lr-lm)).sum(-1)+(pn*(ln-lm)).sum(-1)),.5*np.abs(pr-pn).sum(-1)],-1)

def raw_features(r,n,m):
 lr=r.double().log_softmax(-1);ln=n.double().log_softmax(-1);pr=lr.exp();pn=ln.exp();lm=torch.logaddexp(lr,ln)-__import__('math').log(2.)
 tr=lr.topk(2,dim=-1).values;tn=ln.topk(2,dim=-1).values
 stats=torch.stack([-(pr*lr).sum(-1),-(pn*ln).sum(-1),tr[:,0]-tr[:,1],tn[:,0]-tn[:,1],(pr*(lr-ln)).sum(-1),(pn*(ln-lr)).sum(-1),.5*((pr*(lr-lm)).sum(-1)+(pn*(ln-lm)).sum(-1)),.5*(pr-pn).abs().sum(-1)],-1)
 target=torch.as_tensor(m['ids'],device=D)[:,None]
 rp=r.log_softmax(-1).gather(1,target).squeeze(1);np_=n.log_softmax(-1).gather(1,target).squeeze(1)
 owner=torch.as_tensor(m['owner'],device=D);a=[];b=[]
 for wi in range(m['word_count']):
  mask=owner==wi;rs=rp[mask].double();ns=np_[mask].double()
  ra=torch.stack([rs.sum(),rs.mean(),rs.min(),rs[0]]);na=torch.stack([ns.sum(),ns.mean(),ns.min(),ns[0]])
  # Original archived E13 aggregates real/null separately in float64 before subtraction.
  f13=torch.cat([ra,na,ra-na,ra.new_tensor([float(mask.sum())])]).float()
  z=stats[mask];f32=torch.cat([z.sum(0),z.mean(0),z.min(0).values,z[0]]).float()
  a.append(f13);b.append(f32)
 a=F.pad(torch.stack(a),(0,0,0,12-m['word_count']));b=F.pad(torch.stack(b),(0,0,0,12-m['word_count']))
 return torch.cat([a,b],-1),torch.stack([rp,np_],-1)

E13_NAMES=['real_sum','real_mean','real_min','real_first','null_sum','null_mean','null_min','null_first','difference_sum','difference_mean','difference_min','difference_first','subword_count']
STAT_NAMES=['entropy_real','entropy_null','top2_gap_real','top2_gap_null','KL_real_null','KL_null_real','JS','TV']
FEATURE_NAMES=E13_NAMES+[a+'.'+s for a in ['sum','mean','min','first'] for s in STAT_NAMES]
def pack45(lp,stats,owner,word_count):
 assert lp.shape==(len(owner),2) and stats.shape==(len(owner),8)
 result=np.zeros((12,45),np.float32)
 for wi in range(word_count):
  mask=owner==wi;pair=lp[mask].astype(np.float64);assert len(pair)
  def agg(v):return np.array([v.sum(),v.mean(),v.min(),v[0]],np.float64)
  r,n=agg(pair[:,0]),agg(pair[:,1]);v=stats[mask]
  result[wi]=np.r_[r,n,r-n,len(pair),v.sum(0),v.mean(0),v.min(0),v[0]].astype(np.float32)
 assert np.isfinite(result).all()
 return result

def marker_token_ids(tokenizer):
 ids=[]
 for tok,i in tokenizer.get_vocab().items():
  i=int(i)
  if not 0<=i<151936:continue
  value=tokenizer.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False).strip().casefold()
  if re.fullmatch(r'x+',value):ids.append(i)
 ids=sorted(set(ids));assert len(ids)>=2
 return np.asarray(ids,np.int64)
def marker_logmass(r,n,indices):
 ii=torch.as_tensor(indices,device=r.device)
 def value(a):
  a=a.double();return torch.logsumexp(a[:,ii],-1)-torch.logsumexp(a,-1)
 out=torch.stack([value(r),value(n)],-1).cpu().numpy()
 assert np.isfinite(out).all() and out.max()<=1e-9
 return out
def marker_global6(logmass,owner,word_count):
 first=np.asarray([np.flatnonzero(owner==j)[0] for j in range(word_count)],np.int64)
 z=logmass[first];delta=z[:,0]-z[:,1]
 return np.asarray([logmass[0,0],logmass[0,1],logmass[0,0]-logmass[0,1],z[:,0].mean(),z[:,1].mean(),delta.mean()],np.float32)

def cpu_marker_logmass(r,n,indices):
 def lse(v):
  v=v.astype(np.float64);mx=v.max(-1,keepdims=True)
  return mx[:,0]+np.log(np.exp(v-mx).sum(-1))
 return np.stack([lse(r[:,indices])-lse(r),lse(n[:,indices])-lse(n)],-1)
