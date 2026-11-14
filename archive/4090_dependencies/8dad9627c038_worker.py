"""Frozen LMY word-gated MVDR measurement on public EVAL audio."""
import argparse,ast,copy,hashlib,io,json,math,pathlib,re,sys,time,types
from functools import lru_cache

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();root=pathlib.Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 sys.path.insert(0,cfg['vendor'])
 import numpy as np,torch,soundfile as sf,soxr
 from torch.nn import functional as F
 from scipy import signal
 from scipy.special import logit,xlogy
 from safetensors import safe_open
 from tokenizers import Tokenizer
 from transformers import AutoModelForCTC,AutoProcessor
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.enabled=False;torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,indent=2,allow_nan=False));p.replace(out/n)
 def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 def definitions(item,env):
  p=pathlib.Path(item['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['SHA'];nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in item['names']];assert{n.name for n in nodes}==set(item['names']);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 rows=json.loads((root/'selection.json').read_text());assert len(rows)==2191 and all(r['role']=='EVAL'for r in rows[2:]);ids=[r['signal']for r in rows];assert len(ids)==len(set(ids));assets={r['signal']:dict(words=r['prompt'].split(),files=dict(audio1=dict(path=r['audio'],sha256=r['audio_SHA'])))for r in rows}
 env=dict(np=np,torch=torch,F=F,sf=sf,soxr=soxr,signal=signal,logit=logit,xlogy=xlogy,pathlib=pathlib,hashlib=hashlib,io=io,math=math,re=re,lru_cache=lru_cache,FS=16000,NFFT=512,HOP=128,PSEUDOCOUNT=2.,assets=assets)
 for item in cfg['alignment_definitions']:definitions(item,env)
 snapshot=pathlib.Path(cfg['native']['snapshot']);tok=Tokenizer.from_file(str(snapshot/'tokenizer.json'))
 with safe_open(str(snapshot/'model.safetensors'),framework='pt',device='cpu')as f:W=f.get_tensor('encoder.out.weight').float().cuda();bias=f.get_tensor('encoder.out.bias').float().cuda()
 caches={}
 for role,p in cfg['caches'].items():
  p=pathlib.Path(p);ix=json.loads((p/'index.json').read_text());caches[role]=(ix,np.memmap(p/'features.f16',mode='r',dtype=np.float16,shape=(ix['total_tokens'],1024)),{s:i for i,s in enumerate(ix['signal_ids'])})
 weights=[];offsets=[0];proof=[]
 with torch.no_grad():
  for i,r in enumerate(rows):
   ix,mm,lookup=caches['VALID'if i<2 else'EVAL'];j=lookup[r['signal']];o=ix['offsets'][j];T=ix['lengths'][j];frames=np.asarray(mm[o:o+T],dtype=np.float32).copy();tokens,owner=env['token_word_map'](assets[r['signal']]['words'],tok);lp=F.linear(torch.from_numpy(frames).cuda(),W,bias).double().log_softmax(-1);ga,Z=env['occupancy'](lp[:,tokens].cpu().numpy(),lp[:,0].cpu().numpy(),tokens);w=np.zeros((T,12),np.float32)
   for k in range(r['n_words']):
    pos=np.flatnonzero(owner==k)
    if ga is not None and len(pos)and ga[:,pos].sum()>0:pp=ga[:,pos].sum(1);w[:,k]=pp/pp.sum()
    else:w[:,k]=1/T
   assert np.isfinite(w).all()and abs(w[:,:r['n_words']].sum(0)-1).max()<2e-6
   if i<2 and ga is not None:
    dense=float(F.ctc_loss(lp[:,None,:].cpu(),torch.tensor(tokens),torch.tensor([T]),torch.tensor([len(tokens)]),blank=0,reduction='sum'));assert abs(dense+Z)<1e-8;proof.append(dict(signal=r['signal'],dense_ctc_gap=abs(dense+Z)))
   weights.append(w);offsets.append(offsets[-1]+T)
 env.update(al=dict(weights=np.concatenate(weights),offsets=np.asarray(offsets)),idx={s:i for i,s in enumerate(ids)})
 np.savez_compressed(out/'PUBLIC_ALIGNMENT.npz',signals=np.asarray(ids),**env['al']);del lp,frames;torch.cuda.empty_cache();put('alignment_complete.json',dict(rows=len(rows),control=proof))
 # The original native operator uses the frozen response head and FP16 encoder.
 processor=AutoProcessor.from_pretrained(str(snapshot),trust_remote_code=False,local_files_only=True);pm,info=AutoModelForCTC.from_pretrained(str(snapshot),dtype=torch.float16,trust_remote_code=False,local_files_only=True,output_loading_info=True);assert not any(info.get(k)for k in ['missing_keys','mismatched_keys','error_msgs']);assert len(pm.encoder.layers)==16
 base_layer=copy.deepcopy(pm.encoder.layers[-1]).cuda().eval().requires_grad_(False);position=pm.encoder.attention_dists.detach().clone().cuda();pm.encoder.layers=pm.encoder.layers[:-1];pm.cuda().eval().requires_grad_(False);cp=torch.load(cfg['native']['warm_checkpoint'],map_location='cpu',weights_only=False);WA=cp['weights']['response.A'].float().cuda();WB=cp['weights']['response.B'].float().cuda();del cp
 for q in [W,bias,WA,WB]:q.requires_grad_(False)
 event=types.ModuleType('frozen_word_event');p=pathlib.Path(cfg['event']['path']);assert sha(p)==cfg['event']['SHA'];exec(compile(p.read_text(),str(p),'exec'),event.__dict__)
 env.update({k:getattr(event,k)for k in ['tokenizer_pieces','piece_description','compile_word','compress_prob','forward_event']});pieces=event.tokenizer_pieces(snapshot/'tokenizer.json');env.update(pieces=pieces,descriptions=[event.piece_description(p)for p in pieces],pm=pm,processor=processor,base_layer=base_layer,position=position,W=W,bias=bias,WA=WA,WB=WB)
 for item in cfg['measurement_definitions']:definitions(item,env)
 hd=types.ModuleType('frozen_mvdr_hdemucs');sys.modules[hd.__name__]=hd;p=pathlib.Path(cfg['separator_source']);exec(compile(p.read_text(),str(p),'exec'),hd.__dict__);model=hd.hdemucs_high(sources=['drums','bass','other','vocals']);model.load_state_dict(torch.load(cfg['separator_weight'],map_location='cpu',weights_only=True),strict=True);model.cuda().eval().requires_grad_(False);env['model']=model;vi=list(model.sources).index('vocals');definitions(cfg['separate_definition'],env)
 with np.load(cfg['valid_features'],allow_pickle=False)as z:target=z['features'][[z['signals'].tolist().index(s)for s in ids[:2]]].copy()
 result=[];controls=[];previous_key=None;previous_value=None
 for i,r in enumerate(rows):
  s=r['signal'];nw=r['n_words'];key=(r['audio_SHA'],tuple(assets[s]['words']));x=env['stereo'](s)
  if i>=2 and key==previous_key:v=previous_value.copy()
  else:
   raw,sr=sf.read(r['audio'],dtype='float32',always_2d=True)
   if sr!=44100:raw=soxr.resample(raw,sr,44100,quality='HQ').astype(np.float32)
   y=torch.from_numpy(raw.T.copy());mu=y.mean();scale=y.mean(0).std().clamp_min(1e-8);stems=env['separate']((y-mu)/scale);vocal=(stems[vi]*scale).mean(0).numpy();vocal=soxr.resample(vocal,44100,16000,quality='HQ').astype(np.float32);mix=env['audio'](s);assert len(x)==len(mix)==len(vocal)
   raw6=env['profiles'](s,env['probability_audio'](np.ascontiguousarray(mix,dtype=np.float32)));globalwave,_=env['resynthesize'](x,mix,vocal);ps=[env['profiles'](s,env['probability_audio'](np.ascontiguousarray(globalwave[:,ear],dtype=np.float32)))for ear in [0,1]];global12=env['sym'](*ps);T=env['stft'](mix).shape[1];gate,_=env['gates'](s,len(x),T);localwave,_=env['resynthesize_word_gated'](x,mix,vocal,gate);ps=[env['profiles'](s,env['probability_audio'](np.ascontiguousarray(localwave[:,ear],dtype=np.float32)))for ear in [0,1]];local12=env['sym'](*ps);v=np.c_[raw6,global12,global12,local12-global12];v[nw:]=0;previous_key=key;previous_value=v.copy()
  assert v.shape==(12,42)and np.isfinite(v).all()
  if i<2:
   gap=float(abs(v-target[i]).max());assert gap<.002,(s,gap);controls.append(dict(signal=s,maxgap=gap))
   if i==1:put('VALID_control_passed.json',dict(cases=controls))
  else:result.append(v)
  if i%8==0 or i==len(rows)-1:
   progress=dict(done=max(0,i-1),total=2189,seconds=time.monotonic()-start);put('progress.json',progress);print(json.dumps(progress),flush=True)
   if result:np.savez_compressed(out/'EVAL_MVDR_partial.npz',signals=np.asarray(ids[2:i+1]),features=np.asarray(result))
 p=out/'EVAL_MVDR_FEATURES.npz';(out/'EVAL_MVDR_partial.npz').replace(p);put('complete.json',dict(rows=2189,training_updates=0,VALID_controls=controls,seconds=time.monotonic()-start,output_SHA=sha(p)))

if __name__=='__main__':main()
