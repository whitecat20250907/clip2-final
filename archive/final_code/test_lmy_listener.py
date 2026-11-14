"""Unchanged Whisper-medium full_newFIT normalized K8 fixed inference."""
import argparse,ast,hashlib,json,math,pathlib,re,sys,time,types,unicodedata
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();root=pathlib.Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);(out/'rows').mkdir(exist_ok=True);start=time.monotonic()
 sys.path.insert(0,str(root/'tf_overlay'))
 import numpy as np,torch,soundfile as sf
 from torch import nn
 import transformers
 assert torch.__version__==cfg.get('torch','2.5.1+cu121')and transformers.__version__=='5.14.1'
 import transformers.utils.import_utils as availability
 availability._torchvision_available=False;availability._torchaudio_available=False
 availability.is_torchvision_available=lambda:False;availability.is_torchaudio_available=lambda:False
 transformers.utils.is_torchvision_available=lambda:False;transformers.utils.is_torchaudio_available=lambda:False
 torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,indent=2,allow_nan=False));p.replace(out/n)
 def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 def module(key,names=None):
  item=cfg['sources'][key];p=pathlib.Path(item['path']);assert sha(p)==item['SHA'];env=dict(np=np,torch=torch,nn=nn,math=math,re=re,unicodedata=unicodedata,hashlib=hashlib)
  nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and(names is None or n.name in names)]
  if names is not None:assert {n.name for n in nodes}==set(names)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env);return types.SimpleNamespace(**env)
 lora=module('lora',['LoRALinear','attach']);public=module('public',['normalize','aligned_hits','mapping']);labels=module('labels',['expanded_words']);recipe=module('recipe',['matched_candidates','aggregate','paired_full_inference'])
 env=dict(torch=torch,attach=lora.attach);p=pathlib.Path(cfg['sources']['models']['path']);assert sha(p)==cfg['sources']['models']['SHA'];nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='models'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 base,adapted,tokenizer,extractor=env['models'](pathlib.Path(cfg['model']),device);cp=torch.load(cfg['adapter'],map_location='cpu',weights_only=True)['adapters'];params={n:p for n,p in adapted.named_parameters()if p.requires_grad};assert set(params)==set(cp)
 with torch.no_grad():
  for n,p in params.items():p.copy_(cp[n].to(p.device))
 adapted.eval();contractions=json.loads(pathlib.Path(cfg['asset_protocol']).read_text())['contractions'];rows=json.loads((root/'selection.json').read_text());assert len(rows)==2191
 with np.load(cfg['valid_reference'],allow_pickle=False)as z:
  ids=z['signals'].tolist();offset=np.r_[0,np.cumsum(z['lengths'])];target={s:z['adapted_word3'][offset[j]:offset[j+1]].copy()for j,s in enumerate(ids[:2])}
 proof=[];values=[];lengths=[]
 for i,r in enumerate(rows):
  p=out/'rows'/(r['signal']+'.npz')
  if p.exists():
   with np.load(p,allow_pickle=False)as z:assert str(z['audio_SHA'])==r['audio_SHA'];v=z['adapted_word3'].copy()
  else:
   assert sha(r['audio'])==r['audio_SHA'];wave,sr=sf.read(r['audio'],dtype='float32',always_2d=True);res=recipe.paired_full_inference(base,adapted,tokenizer,extractor,wave,sr,r['signal'],r['prompt'],device,public,labels,contractions);v=res['adapted']['normalized']['word3'];assert v.shape==(r['n_words'],3)and np.isfinite(v).all()
   np.savez_compressed(p,signal=r['signal'],audio_SHA=r['audio_SHA'],adapted_word3=v,teacher2=res['adapted']['teacher_by_ear'],base_word3=res['base']['normalized']['word3']);put('row_'+r['signal']+'.json',dict(adapted_texts=res['adapted']['texts_by_ear'],base_texts=res['base']['texts_by_ear']))
  if i<2:
   gap=float(abs(v-target[r['signal']]).max());proof.append(dict(signal=r['signal'],maxgap=gap));put('VALID_control_observations.json',dict(cases=proof));assert gap==0,proof
   if i==1:put('VALID_control_passed.json',dict(cases=proof))
  else:values.append(v);lengths.append(len(v))
  if i%8==0 or i==len(rows)-1:put('progress.json',dict(done=max(0,i-1),total=2189,seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/2**20));print('DONE',max(0,i-1),flush=True)
 assert all(torch.equal(p.detach().cpu(),cp[n])for n,p in params.items());p=out/'EVAL_LISTENER_WORD3.npz';np.savez_compressed(p,signals=np.array([r['signal']for r in rows[2:]]),lengths=np.array(lengths),adapted_word3=np.concatenate(values));put('complete.json',dict(rows=2189,training_updates=0,parameters_unchanged=True,VALID_controls=proof,output_SHA=sha(p),seconds=time.monotonic()-start))
if __name__=='__main__':main()
