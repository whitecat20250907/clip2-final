"""Original fixed two-fold Qwen 1.7B greedy ASR and nine word features."""
import argparse,ast,gc,hashlib,json,pathlib,re,sys,time,unicodedata

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args()
 root=pathlib.Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);(out/'rows').mkdir(exist_ok=True);start=time.monotonic()
 for p in reversed(cfg['python_paths']):sys.path.insert(0,p)
 import numpy as np,torch,transformers,tokenizers
 import transformers.utils.import_utils as av
 av._torchvision_available=False;av._torchaudio_available=False
 import peft
 from qwen_asr import Qwen3ASRModel
 assert torch.__version__=='2.9.1+cu128'and transformers.__version__=='4.57.6'and peft.__version__=='0.18.1'and tokenizers.__version__=='0.22.2'
 torch.set_num_threads(2);torch.manual_seed(20261017);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def sha(p):
  h=hashlib.sha256()
  with pathlib.Path(p).open('rb')as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  return h.hexdigest()
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False));p.replace(out/n)
 env=dict(np=np,re=re,unicodedata=unicodedata)
 for key,names in [('alignment',['_words','align_words']),('features',['normalize_machine','char_distance','raw_features'])]:
  p=pathlib.Path(cfg[key]['path']);assert sha(p)==cfg[key]['SHA'];nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert{n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 rows=json.loads((root/'selection.json').read_text());controls=[r for r in rows if r['role']=='VALID_CONTROL'];evaluation=[r for r in rows if r['role']=='EVAL'];assert len(evaluation)==2189
 expected={v['signal']:v for v in [json.loads(l)for l in pathlib.Path(cfg['reference_transcripts']).read_text().splitlines()if l]}
 norm=json.loads(pathlib.Path(cfg['normalization']).read_text());mu=np.asarray(norm['mean']);sd=np.asarray(norm['std']);assert mu.shape==sd.shape==(9,)
 proofs=[]
 for fold in [0,1]:
  source=cfg['teachers'][str(fold)];cp=pathlib.Path(source['path']);assert sha(cp/'adapter_model.safetensors')==source['SHA']
  put('progress.json',dict(stage='loading_fixed_fold',fold=fold,seconds=time.monotonic()-start))
  wrapper=Qwen3ASRModel.from_pretrained(cfg['model'],dtype=torch.bfloat16,device_map=None,max_inference_batch_size=8,max_new_tokens=128)
  wrapper.model=peft.PeftModel.from_pretrained(wrapper.model,str(cp)).merge_and_unload();wrapper.model.to('cuda').eval().requires_grad_(False);wrapper.model.generation_config.do_sample=False
  for section in [controls,evaluation]:
   selected=[r for r in section if r['fold']==fold]
   for lo in range(0,len(selected),8):
    group=selected[lo:lo+8];missing=[r for r in group if not(out/'rows'/(r['signal']+'.json')).exists()]
    if missing:
     # Preserve the original batch composition on resume, including completed rows.
     for r in group:assert sha(r['audio'])==r['audio_SHA']
     with torch.inference_mode():results=wrapper.transcribe(audio=[r['audio']for r in group],language='English')
     assert len(results)==len(group)
     for r,item in zip(group,results):
      result=dict(signal=r['signal'],text=str(item.text),fold=fold,audio_SHA=r['audio_SHA']);p=out/'rows'/(r['signal']+'.json')
      if p.exists():assert json.loads(p.read_text())==result
      else:p.write_text(json.dumps(result,ensure_ascii=False))
    if section is controls:
     for r in group:
      actual=json.loads((out/'rows'/(r['signal']+'.json')).read_text());old=expected[r['signal']];assert old['fold']==fold
      v=env['raw_features'](r['prompt'],actual['text']);reference=env['raw_features'](r['prompt'],old['text']);gap=float(abs(v-reference).max())
      proof=dict(signal=r['signal'],fold=fold,actual_text=actual['text'],archived_text=old['text'],raw9_gap=gap);proofs.append(proof);put('VALID_control_observations.json',dict(cases=proofs));assert gap==0,proof
    done=sum((out/'rows'/(r['signal']+'.json')).exists()for r in evaluation);put('progress.json',dict(stage='fixed_ASR',fold=fold,done=done,total=2189,seconds=time.monotonic()-start));print('DONE',done,flush=True)
  assert sha(cp/'adapter_model.safetensors')==source['SHA'];del wrapper;gc.collect();torch.cuda.empty_cache()
 put('VALID_control_passed.json',dict(cases=proofs))
 raw=np.zeros((len(rows),12,9),np.float32);available=np.zeros((len(rows),12),bool);wordmask=np.zeros_like(available);texts=[]
 for i,r in enumerate(rows):
  pred=json.loads((out/'rows'/(r['signal']+'.json')).read_text());assert pred['audio_SHA']==r['audio_SHA']and pred['fold']==r['fold'];texts.append(pred['text']);n=r['n_words'];v=env['raw_features'](r['prompt'],pred['text']);assert v.shape==(n,9);raw[i,:n]=v;wordmask[i,:n]=True
  align=env['align_words'](r['prompt'],env['normalize_machine'](pred['text']));available[i,:n]=[h is not None and unique for h,unique in zip(align['hyp_indices'],align['mapping_mask'])]
 normalized=np.clip((raw-mu)/sd,-6,6).astype(np.float32);normalized[~wordmask]=0
 p=out/'EVAL_FREE_ASR9_AVAILABILITY.npz';np.savez_compressed(p,signals=np.asarray([r['signal']for r in rows]),wordmask=wordmask,raw9=raw,normalized9=normalized,available=available,texts=np.asarray(texts),fold=np.asarray([r['fold']for r in rows]))
 put('complete.json',dict(rows=2189,VALID_control_rows=len(controls),training_updates=0,output_SHA=sha(p),VALID_controls=proofs,seconds=time.monotonic()-start))

if __name__=='__main__':main()
