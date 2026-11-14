"""Exact frozen Qwen prefix moments and band24; two VALID controls first."""
import argparse,ast,hashlib,importlib.util,json,math,os,pathlib,re,sys,time,types
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=pathlib.Path(a.base);c=json.loads((b/'protocol.json').read_text());out=b/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 sys.path.insert(0,c['runtime_site'])
 import numpy as np,torch,soundfile as sf,soxr,transformers
 from scipy.special import rel_entr
 from torch import nn
 from torch.nn import functional as F
 from safetensors import safe_open
 import transformers.utils.import_utils as av
 av._torchvision_available=False;torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest');free,total=torch.cuda.mem_get_info();assert free>5*1024**3;torch.cuda.set_per_process_memory_fraction(4*1024**3/total)
 env=dict(np=np,torch=torch,sf=sf,soxr=soxr,transformers=transformers,nn=nn,F=F,safe_open=safe_open,os=os,sys=sys,pathlib=pathlib,json=json,hashlib=hashlib,math=math,types=types,importlib=__import__('importlib'),rel_entr=rel_entr,Q=pathlib.Path(c['runtime_root']),MODEL=pathlib.Path(c['model']))
 def defs(name,names):
  p=pathlib.Path(c['sources'][name]['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==c['sources'][name]['SHA'];nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert{n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 defs('common_v1.py',['sha','read','mod','imports','own_mask','mask_hook','load_tower','prepare','forward','LoRA','insert_lora','ctc_logp','ctc'])
 defs('prefix_common_v1.py',['PrefixCut','Prefix','tail','mathctx']);defs('score_math_v1.py',['greedy','edits','minimum']);defs('study_common_v1.py',['public_hash','tensor_hash','set_delta','features','disagreement'])
 defs('micro.py',['getmel']);defs('band.py',['band','fromcache','fromwave','make24'])
 rows=json.loads((b/'selection.json').read_text());assert len(rows)==2191 and all(r['role']=='EVAL' for r in rows[2:]);assert not any({'response','correctness','target','labels'}&r.keys()for r in rows)
 phones={p:i+1 for i,p in enumerate('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())};dictionary={}
 for line in pathlib.Path(c['dictionary']).read_text().splitlines():
  fields=line.split()
  if len(fields)<2 or fields[0].startswith(';;;'):continue
  word=re.sub(r'\(\d+\)$','',fields[0]).lower();seq=[re.sub(r'\d','',p)for p in fields[1:]]
  if word not in dictionary and all(p in phones for p in seq):dictionary[word]=[phones[p]for p in seq]
 records={r['signal']:dict(n_words=r['n_words'],severity=r['severity'],reference=None if any(w not in dictionary for w in r['prompt'].split())else[p for w in r['prompt'].split()for p in dictionary[w]])for r in rows}
 tower,extractor,_,_=env['load_tower']();tower.to('cuda:0');tower.config._attn_implementation='sdpa';original=env['public_hash'](tower);mods=env['insert_lora'](tower);cp=torch.load(c['seeds']['44']['checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==945 and cp['seed']==44 and cp['public_frozen_parameter_sha']==original;env['set_delta'](mods,cp['audio_delta'])
 heads=nn.ModuleDict({k:nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,64),nn.GELU(),nn.Linear(64,40))for k in ['ref','response']});heads.load_state_dict(cp['heads']);heads.to('cuda:0').eval().requires_grad_(False);tower.eval().requires_grad_(False);capture=env['Prefix'](tower)
 old39={}
 for file,field in [(c['seeds']['44']['validation_q39'],'features'),(c['eval_q39'],'q39')]:
  with np.load(file,allow_pickle=False)as z:
   for i,s in enumerate(z['signals'].tolist()):old39[s]=z[field][i].copy()
 env.update(tower=tower,extractor=extractor,heads=heads,records=records,old39=old39,cols=np.asarray([0,3,4,5,6,7,8,9,10]))
 with np.load(c['validation_features'],allow_pickle=False)as z:expected={r['signal']:{k:z[k][z['signals'].tolist().index(r['signal'])].copy()for k in ['h14_RP128','band24','audio2_abs128']}for r in rows[:2]}
 P=(np.random.default_rng(20261020).standard_normal((896,64))/np.sqrt(896)).astype(np.float32)
 def pool(h):
  rp=(h@P).astype(np.float64);return np.r_[rp.mean(0),rp.std(0)].astype(np.float32)
 def wave(path):
  aa,sr=sf.read(str(path),dtype='float32',always_2d=True);assert aa.shape[1]==2 and np.isfinite(aa).all();w=aa.mean(1,dtype=np.float32)
  if sr!=16000:w=soxr.resample(w,sr,16000,quality='HQ').astype(np.float32)
  return w
 proofs=[];results=[];a2memo={}
 with torch.no_grad(),env['mathctx']():
  for i,r in enumerate(rows):
   if i==2:
    assert len(proofs)==2;(out/'VALID_control_passed.json').write_text(json.dumps(proofs,indent=2))
   w=wave(r['audio']);mel,_=env['getmel'](w);h,cu=capture(mel.to('cuda:0'));h=h.numpy();cu=cu.numpy();base=env['fromcache'](h,cu);band=env['fromwave'](env['band'](w));v=dict(h14_RP128=pool(h),band24=env['make24'](r['signal'],base,band))
   path=pathlib.Path(r['audio2']);key=env['sha'](path)
   if key not in a2memo:
    m,_=env['getmel'](wave(path));h2,_=capture(m.to('cuda:0'));a2memo[key]=pool(h2.numpy())
   v['audio2_abs128']=a2memo[key]
   if i<2:
    gaps={}
    for k,q in v.items():
     target=expected[r['signal']][k];assert np.array_equal(np.isnan(q),np.isnan(target));gaps[k]=float(np.nanmax(abs(q-target)));assert gaps[k]<2e-5,(r['signal'],k,gaps[k])
    proofs.append(dict(signal=r['signal'],maxgaps=gaps))
   else:results.append(v)
   if i%64==0 or i==len(rows)-1:
    p=dict(done=max(0,i-1),total=2189,seconds=time.monotonic()-start,audio2_unique=len(a2memo));(out/'progress.json').write_text(json.dumps(p));print(json.dumps(p),flush=True)
 capture.hook.remove();assert env['public_hash'](tower)==original
 data={k:np.asarray([r[k]for r in results])for k in results[0]};assert all(not np.isinf(v).any()for v in data.values())
 p=out/'EVAL_PREFIX_BAND280.npz'
 with p.open('xb')as f:np.savez_compressed(f,signals=np.asarray([r['signal']for r in rows[2:]]),**data)
 (out/'complete.json').write_text(json.dumps(dict(rows=2189,training_updates=0,VALID_controls=proofs,seconds=time.monotonic()-start,output_SHA=env['sha'](p)),indent=2))
if __name__=='__main__':main()
