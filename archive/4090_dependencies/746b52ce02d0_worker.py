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
 rows=json.loads((root/'selection.json').read_text());assert len(rows)==2 and all(x['role']=='VALID_CONTROL'for x in rows)
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
 flags=dict(flash=torch.backends.cuda.flash_sdp_enabled(),memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled(),cudnn=torch.backends.cuda.cudnn_sdp_enabled(),bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
 variants=[('default',False,False),('no_bf16_reduced_accumulation',False,True),('math_attention',True,False),('math_attention_no_bf16_reduced_accumulation',True,True),('default_repeat',False,False)]
 reports=[];baseline={}
 for name,math_only,no_reduce in variants:
  torch.backends.cuda.enable_flash_sdp(False if math_only else flags['flash']);torch.backends.cuda.enable_mem_efficient_sdp(False if math_only else flags['memory_efficient']);torch.backends.cuda.enable_cudnn_sdp(False if math_only else flags['cudnn']);torch.backends.cuda.enable_math_sdp(True if math_only else flags['math']);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False if no_reduce else flags['bf16_reduced']
  for i,row in enumerate(rows):
   rr=dict(signal=row['signal'],text=row['prompt'],words=row['n_words'],audio=row['audio']);x,m,h=env['prepare'](rr);assert h==row['audio_SHA'];captured=[]
   hook=native.thinker.audio_tower.register_forward_hook(lambda mod,args,o:captured.append(o.last_hidden_state.detach().cpu().to(torch.float16).clone()))
   try:r,rp=env['distribution_logits'](x,m,False)
   finally:hook.remove()
   n,np_=env['distribution_logits'](x,m,True);stats=env['gpu_stats'](r,n);lp=np.stack([rp,np_],-1);raw=env['pack45'](lp,stats,m['owner'],m['word_count']);legal=np.arange(12)<row['n_words'];x45=np.clip((raw-mu)/sd,-6,6).astype(np.float32);x45[~legal]=0;frames=captured[0].numpy()
   if name=='default':baseline[row['signal']]=dict(x45=x45.copy(),frames=frames.copy())
   old=baseline[row['signal']];dif=x45.astype(np.float64)-expected[i]
   rec=dict(variant=name,signal=row['signal'],normalized_maxgap_reference=float(abs(dif).max()),normalized_RMS_reference=float(np.sqrt(np.mean(dif[legal]**2))),normalized_maxgap_default=float(abs(x45-old['x45']).max()),frames_maxgap_default=float(abs(frames.astype(np.float32)-old['frames'].astype(np.float32)).max()),same_default_frames=bool(np.array_equal(frames,old['frames'])),math_attention=math_only,bf16_reduced_accumulation=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
   reports.append(rec);np.savez_compressed(out/(name+'_'+row['signal']+'.npz'),signal=row['signal'],text=row['prompt'],audio_SHA=h,raw45=raw,x45=x45,frames=frames,token_logp=lp);put('progress.json',dict(done=len(reports),total=len(variants)*len(rows),latest=rec));print(json.dumps(rec),flush=True);del r,n,x;torch.cuda.empty_cache()
 assert adapters_exact()and env['base_hash']()==expected_base
 put('complete.json',dict(cases=reports,rows=2,variants=[v[0]for v in variants],default_backend_flags=flags,torch=torch.__version__,transformers=transformers.__version__,GPU=torch.cuda.get_device_name(0),training_updates=0,parameters_unchanged=True,diagnostic_only=True,production_configuration_changed=False,seconds=time.monotonic()-start))

if __name__=='__main__':main()
