"""Frozen LMY TDT word32 and global50 features; no optimization or labels."""
import argparse,ast,hashlib,json,math,pathlib,re,sys,time
from functools import lru_cache

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();root=pathlib.Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 import numpy as np,torch
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def read(p):return json.loads(pathlib.Path(p).read_text())
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,indent=2,allow_nan=False));p.replace(out/n)
 for path in cfg['module_paths']:sys.path.insert(0,path)
 from native_tdt_core_v1 import NativeTDT
 from tdt_loss_core_v1 import LowRankJoint
 from aligned_posterior_core_v1 import extract32
 from projection_core_v1 import ProjectionPair,AcousticWrapper,ContextWrapper
 env=dict(np=np,torch=torch,re=re,math=math,lru_cache=lru_cache)
 for item in cfg['definitions']:
  p=pathlib.Path(item['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['SHA'];nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in item['names']];assert{n.name for n in nodes}==set(item['names']);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 dictionary={}
 for line in pathlib.Path(cfg['cmu']).read_text().splitlines():
  f=line.split()
  if len(f)<2 or f[0].startswith(';;;'):continue
  w=re.sub(r'\(\d+\)$','',f[0]).lower();v=tuple(re.sub(r'\d','',z)for z in f[1:])
  if v not in dictionary.setdefault(w,[]):dictionary[w].append(v)
 env['dictionary']=dictionary
 engine=NativeTDT(cfg['native_plan']);assert engine.weight_sha=='a7829b1bf4118835e354c45965e8e579443adc5731a08f7f6ffacba74a861cbb'
 base=engine.net['joint'].head;W=base.weight.detach();bias=base.bias.detach();human=LowRankJoint(W,bias,84044);cp=torch.load(cfg['human_checkpoint'],map_location='cuda',weights_only=False);assert cp['steps']==768
 with torch.no_grad():
  for k in ['A','B']:getattr(human,k).copy_(cp['heads']['human'][k])
 human.requires_grad_(False).eval();rc=torch.load(cfg['route_checkpoint'],map_location='cuda',weights_only=False);assert rc['steps']==1260;heads={}
 for sev in range(3):
  h=LowRankJoint(W,bias,84044+sev)
  with torch.no_grad():
   for k in ['A','B']:getattr(h,k).copy_(rc['heads'][str(sev)][k])
  heads[sev]=h.requires_grad_(False).eval()
 engine2=NativeTDT(cfg['native_plan']);pc=torch.load(cfg['projection_checkpoint'],map_location='cuda',weights_only=False);assert pc['steps']==2520;ad=ProjectionPair();ad.load_state_dict(pc['adapter'],strict=True);ad.requires_grad_(False).eval();engine2.net['encoder_projector']=AcousticWrapper(engine2.net['encoder_projector'],ad);engine2.net['decoder'].decoder_projector=ContextWrapper(engine2.net['decoder'].decoder_projector,ad)
 rows=read(root/'selection.json');assert len(rows)==2191 and all(r['role']=='EVAL'for r in rows[2:]);ids=[r['signal']for r in rows];assert len(set(ids))==len(ids);texts={};legal={}
 for file in [cfg['valid_native'],cfg['eval_native']]:
  with np.load(file,allow_pickle=False)as z:
   for i,s in enumerate(z['signals'].tolist()):texts[s]=z['texts'][i].tolist();legal[s]=z['legal'][i].copy()
 caches={}
 for role,p in cfg['caches'].items():
  p=pathlib.Path(p);ix=read(p/'index.json');assert ix['dim']==1024;caches[role]=(ix,np.memmap(p/'features.f16',mode='r',dtype=np.float16,shape=(ix['total_tokens'],1024)),{s:i for i,s in enumerate(ix['signal_ids'])})
 def frames(r):
  ix,mm,lookup=caches['VALID'if r['role']=='VALID_CONTROL'else'EVAL'];j=lookup[r['signal']];off=ix['offsets'][j];return np.asarray(mm[off:off+ix['lengths'][j]],dtype=np.float32).copy()
 keys=['old32','new32','frozen50','human50']
 with np.load(cfg['valid_tdt'],allow_pickle=False)as z:
  ix=[z['signals'].tolist().index(s)for s in ids[:2]];expected={k:z[k][ix].copy()for k in keys}
 banks={k:np.zeros((len(ids),12,32)if k.endswith('32')else(len(ids),50))for k in keys};proof=[];mass=0.
 with torch.inference_mode():
  for lo in [0]+list(range(2,len(ids),16)):
   hi=2 if lo==0 else min(lo+16,len(ids));group=rows[lo:hi];arrays=[frames(r)for r in group];engine.net['joint'].head=base;fr,_=engine.decode_many(arrays);engine.net['joint'].head=human;hu,_=engine.decode_many(arrays)
   for j,(r,array)in enumerate(zip(group,arrays)):
    i=lo+j;s=r['signal'];words=r['prompt'].split();assert len(words)==r['n_words']==int(legal[s].sum())
    for rec,key in [(fr[j],'frozen50'),(hu[j],'human50')]:
     v=env['new25'](words,rec,texts[s],cfg['contractions']);banks[key][i]=np.r_[v.mean(0),v.std(0)]
    for key,en in [('old32',engine),('new32',engine2)]:
     v,g=extract32(en,heads[r['severity']],array,words);banks[key][i,:len(words)]=v;mass=max(mass,float(g))
    if i<2:
     gaps={k:float(np.max(abs(banks[k][i]-expected[k][i])))for k in keys};assert max(gaps.values())<2e-5,(s,gaps);proof.append(dict(signal=s,maxgaps=gaps))
   if lo==0:put('VALID_control_passed.json',dict(cases=proof))
   progress=dict(done=max(0,hi-2),total=2189,seconds=time.monotonic()-start,massmax=mass);put('progress.json',progress);print(json.dumps(progress),flush=True)
   part=out/'EVAL_TDT_partial.npz'
   with part.open('wb')as f:np.savez_compressed(f,signals=np.asarray(ids[2:hi]),legal=np.asarray([legal[s]for s in ids[2:hi]]),**{k:v[2:hi]for k,v in banks.items()})
 assert mass<1e-6 and all(np.isfinite(v).all()for v in banks.values());part.replace(out/'EVAL_TDT_FEATURES.npz');put('complete.json',dict(rows=2189,training_updates=0,massmax=mass,seconds=time.monotonic()-start,VALID_controls=proof,output_SHA=hashlib.sha256((out/'EVAL_TDT_FEATURES.npz').read_bytes()).hexdigest()))

if __name__=='__main__':main()
