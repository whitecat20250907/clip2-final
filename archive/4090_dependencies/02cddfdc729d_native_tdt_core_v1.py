"""Frozen native TDT decoder on cached Parakeet encoder states; no encoder execution."""
import pathlib,json,hashlib,ast,math,re
from types import SimpleNamespace
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer,AddedToken
ACT2FN={'relu':F.relu}
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def public_classes(f,names):
 nodes=[n for n in ast.parse(pathlib.Path(f).read_text()).body if isinstance(n,ast.ClassDef)and n.name in names];assert {n.name for n in nodes}==set(names)
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0);module=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]));exec(compile(module,str(f),'exec'),globals())
class NativeTDT:
 def __init__(self,p,device='cuda'):
  assert sha(p['modeling_source'])==p['modeling_sha']and sha(p['generation_source'])==p['generation_sha']
  public_classes(p['generation_source'],['ParakeetRNNTDecoderCache']);public_classes(p['modeling_source'],['ParakeetRNNTDecoder','ParakeetRNNTJointNetwork','ParakeetTDTJointNetwork'])
  self.snapshot=pathlib.Path(p['snapshot']);cfg=json.loads((self.snapshot/'config.json').read_text());self.cfg=SimpleNamespace(**cfg);self.V=cfg['vocab_size'];self.blank=cfg['blank_token_id'];self.eos=json.loads((self.snapshot/'generation_config.json').read_text())['eos_token_id']
  assert self.V==8193 and self.blank==8192 and self.eos==3 and cfg['durations']==[0,1,2,3,4]
  self.net=nn.ModuleDict({'encoder_projector':nn.Linear(1024,640),'decoder':ParakeetRNNTDecoder(self.cfg),'joint':ParakeetTDTJointNetwork(self.cfg)})
  state={};h=hashlib.sha256()
  with safe_open(str(self.snapshot/'model.safetensors'),framework='pt',device='cpu')as f:
   keys=sorted(k for k in f.keys()if k.startswith(('decoder.','encoder_projector.','joint.')))
   assert len(keys)==15 and set(keys)==set(self.net.state_dict())
   for k in keys:
    v=f.get_tensor(k);assert v.dtype==torch.float32;state[k]=v;h.update(k.encode());h.update(v.numpy().tobytes())
  self.net.load_state_dict(state,strict=True);self.net.requires_grad_(False);self.net.eval().to(device);self.device=device;self.weight_sha=h.hexdigest();self.weight_keys=keys;self.parameters=sum(v.numel()for v in self.net.parameters())
  assert sha(self.snapshot/'tokenizer.json')==p['tokenizer_sha']and sha(self.snapshot/'tokenizer_config.json')==p['tokenizer_config_sha']
  self.tok=Tokenizer.from_file(str(self.snapshot/'tokenizer.json'));self.ref_tok=Tokenizer.from_file(str(self.snapshot/'tokenizer.json'));self.ref_tok.add_special_tokens([AddedToken('<|endoftext|>',special=True)]);assert self.ref_tok.token_to_id('<|endoftext|>')==self.eos
 def text(self,ids):
  return self.tok.decode([v for v in ids if v!=self.eos],skip_special_tokens=True)
 def decode_many(self,arrays,trace_first=0):
  assert all(a.ndim==2 and a.shape[1]==1024 and len(a)>0 and np.isfinite(a).all()for a in arrays)
  B=len(arrays);lengths=[len(a)for a in arrays];x=nn.utils.rnn.pad_sequence([torch.tensor(a,dtype=torch.float32)for a in arrays],batch_first=True).to(self.device);pool=self.net['encoder_projector'](x);lens=torch.tensor(lengths,device=self.device);frame=torch.zeros(B,dtype=torch.long,device=self.device);done=torch.zeros(B,dtype=torch.bool,device=self.device);prev=torch.full((B,),self.blank,dtype=torch.long,device=self.device);cache=ParakeetRNNTDecoderCache(self.cfg);rows=torch.arange(B,device=self.device);cap=self.cfg.max_symbols_per_step*max(lengths)-1
  records=[dict(raw_tokens=[],durations=[],frames=[],step_info=[])for _ in arrays];traces=[[]for _ in range(trace_first)]
  for step in range(cap):
   if bool(done.all()):break
   active=(~done).cpu().tolist();before=frame.cpu().tolist();prediction=self.net['decoder'](prev[:,None],cache=cache);enc=pool[rows,frame.clamp(max=pool.shape[1]-1)][:,None,:];logits=self.net['joint'](decoder_hidden_states=prediction,encoder_hidden_states=enc).squeeze(1);tokens=logits[:,:self.V].argmax(1);dur=logits[:,self.V:].argmax(1);dur=torch.where((tokens==self.blank)&(dur==0),torch.ones_like(dur),dur)
   lp=logits[:,:self.V].double().log_softmax(-1);prob=lp.exp();entropy=-(prob*lp).sum(1)/math.log(self.V);top=prob.topk(2,dim=1).values;info=torch.stack([lp.gather(1,tokens[:,None]).squeeze(1),entropy,prob[:,self.blank],top[:,0]-top[:,1],dur.double()],dim=1).cpu().numpy();ts=tokens.cpu().tolist();ds=dur.cpu().tolist()
   for j in range(B):
    if not active[j]:continue
    r=records[j];r['raw_tokens'].append(ts[j]);r['durations'].append(ds[j]);r['frames'].append(before[j]);r['step_info'].append(info[j].tolist())
    if j<trace_first:traces[j].append(logits[j].cpu().numpy().copy())
   frame=frame+dur;done=done|(tokens==self.eos)|(frame>=lens);prev=torch.where(done,torch.full_like(tokens,self.blank),tokens)
  ended=done.cpu().tolist()
  for j,r in enumerate(records):
   raw=r['raw_tokens'];text=self.text(raw);assert text==self.ref_tok.decode(raw,skip_special_tokens=True);info=np.asarray(r.pop('step_info'),dtype=np.float64);valid=[i for i,t in enumerate(raw)if t not in [0,2,self.blank,self.eos]and not(self.tok.id_to_token(t)or'').startswith('<|')];r['raw_text']=text;r['feature_text']=re.sub(r'<\|[^|]*\|>',' ',text).strip();r['summary']=dict(mean_valid_token_logp=float(info[valid,0].mean())if valid else 0.,entropy=float(info[:,1].mean()),blank_probability=float(info[:,2].mean()),probability_margin=float(info[:,3].mean()),duration_mean=float(info[:,4].mean()),nonblank_count=sum(t!=self.blank and t!=self.eos for t in raw),valid_token_count=len(valid),EOS=bool(raw and raw[-1]==self.eos),cap_hit=not ended[j],steps=len(raw),encoder_frames=lengths[j])
  return records,traces
 def prefix_reference(self,array,max_steps):
  frames=torch.tensor(array,dtype=torch.float32,device=self.device);history=[self.blank];frame=0;tokens=[];durations=[];positions=[];logits_list=[]
  for step in range(max_steps):
   if frame>=len(array):break
   emb=F.embedding(torch.tensor([history],dtype=torch.long,device=self.device),self.net['decoder'].embedding.weight);hidden,_=self.net['decoder'].lstm(emb);prediction=F.linear(hidden[:,-1],self.net['decoder'].decoder_projector.weight,self.net['decoder'].decoder_projector.bias);enc=F.linear(frames[frame:frame+1],self.net['encoder_projector'].weight,self.net['encoder_projector'].bias);logits=F.linear(F.relu(enc+prediction),self.net['joint'].head.weight,self.net['joint'].head.bias)[0];token=int(logits[:self.V].argmax());duration=int(logits[self.V:].argmax());duration=1 if token==self.blank and duration==0 else duration
   positions.append(frame);tokens.append(token);durations.append(duration);logits_list.append(logits.cpu().numpy().copy());frame+=duration
   if token==self.eos:break
   if token!=self.blank:history.append(token)
  return dict(raw_tokens=tokens,durations=durations,frames=positions),logits_list
