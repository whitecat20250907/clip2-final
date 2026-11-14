"""Original full 1.7B native posterior and adapted audio frames; no fitting."""
import argparse,ast,copy,hashlib,json,os,pathlib,re,sys,time

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--role',required=True);a=ap.parse_args()
 root=pathlib.Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);(out/'rows').mkdir(exist_ok=True);start=time.monotonic()
 for p in reversed(cfg['python_paths']):sys.path.insert(0,p)
 import numpy as np,torch,librosa,transformers,tokenizers
 import transformers.utils.import_utils as available
 available._torchvision_available=False;available._torchaudio_available=False
 import peft
 from qwen_asr import Qwen3ASRModel
 from peft.tuners.lora.layer import LoraLayer
 from safetensors.torch import load_file
 assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.57.6' and peft.__version__=='0.18.1'and tokenizers.__version__=='0.22.2'
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,indent=2,allow_nan=False));p.replace(out/n)
 def sha(p):
  h=hashlib.sha256()
  with pathlib.Path(p).open('rb')as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  return h.hexdigest()
 rows=json.loads((root/'selection.json').read_text());assert len(rows)==2191 and all(x['role']=='VALID_CONTROL'for x in rows[:2])and all(x['role']=='EVAL'for x in rows[2:])
 put('progress.json',dict(stage='loading_original_model',seconds=time.monotonic()-start))
 wrapper=Qwen3ASRModel.from_pretrained(cfg['model'],dtype=torch.bfloat16,device_map=None);native=wrapper.model;processor=wrapper.processor
 prefix=processor.apply_chat_template([[{'role':'system','content':''},{'role':'user','content':[{'type':'audio','audio':None}]}]],add_generation_prompt=True,tokenize=False)[0]+'language English<asr_text>'
 env=dict(np=np,torch=torch,librosa=librosa,re=re,pathlib=pathlib,processor=processor,prefix=prefix,native=native,sha=sha,hashlib=hashlib,D=torch.device('cuda:0'),copy=copy)
 p=pathlib.Path(cfg['original_produce']['path']);assert sha(p)==cfg['original_produce']['SHA'];tree=ast.parse(p.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
 names={'grid','prepare','base_hash','distribution_logits','gpu_stats','cpu_stats'};nodes=[n for n in run.body if isinstance(n,ast.FunctionDef)and n.name in names];assert{n.name for n in nodes}==names;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 p=pathlib.Path(cfg['pack']['path']);assert sha(p)==cfg['pack']['SHA'];nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='pack45'];assert len(nodes)==1;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 expected_base='a9b782d7315ca7099cc595babf6400f492b2625818c6be5354fc18215e7123f3';assert env['base_hash']()==expected_base
 pm=peft.PeftModel.from_pretrained(native,cfg['adapter'],is_trainable=False,autocast_adapter_dtype=True);native=pm.get_base_model();env['native']=native
 source=load_file(str(pathlib.Path(cfg['adapter'])/'adapter_model.safetensors'),device='cpu')
 def adapters_exact():
  cur={n.replace('.default.','.'):p for n,p in pm.named_parameters()if '.lora_A.'in n or '.lora_B.'in n}
  return len(cur)==208 and set(cur)==set(source)and sum(p.numel()for p in cur.values())==2392064 and all(p.dtype==torch.float32 and torch.equal(p.detach().cpu(),source[n])for n,p in cur.items())
 assert adapters_exact()
 for m in native.modules():
  if isinstance(m,LoraLayer):m.enable_adapters(True)
 pm.to('cuda:0').eval().requires_grad_(False)
 norm=json.loads(pathlib.Path(cfg['normalization']).read_text());mu=np.asarray(norm['mean']);sd=np.asarray(norm['std']);assert mu.shape==sd.shape==(45,)
 with np.load(cfg['reference_main_cache'],allow_pickle=False)as z:
  ix={s:i for i,s in enumerate(z['signals'].tolist())};expected=z['x45'][[ix[r['signal']]for r in rows[:2]]].copy()
 values=[];controls=[];fingerprints=[];cached={}
 for i,row in enumerate(rows):
  rr=dict(signal=row['signal'],text=row['prompt'],words=row['n_words'],audio=row['audio']);key=(row['audio_SHA'],row['prompt']);saved=out/'rows'/(row['signal']+'.npz')
  if saved.exists():
   with np.load(saved,allow_pickle=False)as z:
    assert str(z['audio_SHA'])==row['audio_SHA']and str(z['text'])==row['prompt'];raw=z['raw45'].copy();x45=z['x45'].copy()
  else:
   x,m,h=env['prepare'](rr);assert h==row['audio_SHA'];captured=[]
   hook=native.thinker.audio_tower.register_forward_hook(lambda mod,args,o:captured.append(o.last_hidden_state.detach().cpu().to(torch.float16).clone()))
   try:r,rp=env['distribution_logits'](x,m,False)
   finally:hook.remove()
   assert len(captured)==1;frames=captured[0].numpy();assert frames.ndim==2 and frames.shape[1]==2048
   n,np_=env['distribution_logits'](x,m,True);stats=env['gpu_stats'](r,n);lp=np.stack([rp,np_],-1);raw=env['pack45'](lp,stats,m['owner'],m['word_count'])
   if i<2:
    cpu=env['cpu_stats'](r.cpu().numpy(),n.cpu().numpy());gap=float(abs(stats-cpu).max());assert gap<1e-9
   del r,n,x;legal=np.arange(12)<row['n_words'];x45=np.clip((raw-mu)/sd,-6,6).astype(np.float32);x45[~legal]=0
   temp=saved.with_suffix('.next.npz');np.savez_compressed(temp,signal=row['signal'],text=row['prompt'],audio_SHA=h,raw45=raw,x45=x45,frames=frames,token_logp=lp,token_ids=m['ids'],token_to_word=m['owner'],token_stats64=stats);temp.replace(saved)
  if i<2:
   gap=float(abs(x45-expected[i]).max());rec=dict(signal=row['signal'],normalized45_maxgap=gap);controls.append(rec);put('VALID_control_observations.json',dict(cases=controls,limit=cfg['maximum_normalized_feature_gap']))
   rec['strict_numeric_replay_passed']=gap<=cfg['maximum_normalized_feature_gap']
   if i==1:put('VALID_control_cross_device.json',dict(cases=controls,training_updates=0,strict_replay_limit=cfg['maximum_normalized_feature_gap']))
  else:values.append(raw);fingerprints.append(row['audio_SHA'])
  put('progress.json',dict(done=max(0,i-1),total=2189,seconds=time.monotonic()-start,stage='control'if i<2 else'EVAL_native_posterior'))
  if i%16==0:print(json.dumps(dict(done=max(0,i-1),total=2189,seconds=time.monotonic()-start)),flush=True)
 assert adapters_exact()and env['base_hash']()==expected_base
 raw=np.asarray(values);legal=np.arange(12)[None,:]<np.array([r['n_words']for r in rows[2:]])[:,None];x45=np.clip((raw-mu)/sd,-6,6).astype(np.float32);x45[~legal]=0
 p=out/'EVAL_QWEN17_RAW45.npz';np.savez_compressed(p,signals=np.array([r['signal']for r in rows[2:]]),legal=legal,raw45=raw,x45=x45,audio_SHA=np.asarray(fingerprints))
 put('complete.json',dict(rows=2189,training_updates=0,VALID_controls=controls,strict_numeric_replay_passed=all(x['strict_numeric_replay_passed']for x in controls),original_recipe_and_parameters_unchanged=True,execution_contract='identical_frozen_weights_original_recipe_4090',base_unchanged=True,adapters_unchanged=True,output_SHA=sha(p),seconds=time.monotonic()-start))

if __name__=='__main__':main()
