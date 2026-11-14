"""Frozen Whisper reference-token likelihood from original FP32 mono frames."""
import argparse,ast,gc,hashlib,json,pathlib,sys,time

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();root=pathlib.Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);(out/'rows').mkdir(exist_ok=True);start=time.monotonic()
 sys.path.insert(0,cfg['tf_overlay'])
 import numpy as np,torch,librosa,transformers
 import torch.nn.functional as F
 import transformers.utils.import_utils as av
 av._torchvision_available=False;av._torchaudio_available=False
 av.is_torchvision_available=lambda:False;av.is_torchaudio_available=lambda:False
 transformers.utils.is_torchvision_available=lambda:False;transformers.utils.is_torchaudio_available=lambda:False
 from transformers import WhisperProcessor,WhisperModel,WhisperForConditionalGeneration
 from transformers.modeling_outputs import BaseModelOutput
 assert torch.__version__=='2.6.0+cu124'and transformers.__version__=='5.14.1'
 torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def sha(p):
  h=hashlib.sha256()
  with pathlib.Path(p).open('rb')as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  return h.hexdigest()
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,indent=2,allow_nan=False));p.replace(out/n)
 source=pathlib.Path(cfg['core']['path']);assert sha(source)==cfg['core']['SHA'];tree=ast.parse(source.read_text());env=dict(np=np);nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef)and n.name=='summarize_nll')or(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FEATURE_NAMES'for t in n.targets))];assert len(nodes)==2;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
 processor=WhisperProcessor.from_pretrained(cfg['model'],local_files_only=True);processor.tokenizer.set_prefix_tokens(language='english',task='transcribe',predict_timestamps=False)
 full=WhisperModel.from_pretrained(cfg['model'],torch_dtype=torch.float32,local_files_only=True);encoder=full.encoder.to(device).eval().requires_grad_(False);del full
 model=WhisperForConditionalGeneration.from_pretrained(cfg['model'],torch_dtype=torch.float16,local_files_only=True);model.model.encoder=torch.nn.Identity();model.to(device).eval().requires_grad_(False);gc.collect();special=torch.as_tensor(sorted(set(processor.tokenizer.all_special_ids)),device=device)
 rows=json.loads((root/'selection.json').read_text());assert len(rows)==2191
 ref=np.load(root/'valid_reference.npy',allow_pickle=False);manifest=json.loads((root/'valid_manifest.json').read_text());ids=manifest['signal_ids'];index={s:i for i,s in enumerate(ids)}
 @torch.inference_mode()
 def score(frames,prompt):
  labels=processor.tokenizer(prompt,return_tensors='pt').input_ids.to(device);lexical=~torch.isin(labels[0],special);assert lexical.any();output=model(encoder_outputs=BaseModelOutput(last_hidden_state=torch.as_tensor(frames,device=device,dtype=torch.float16)[None]),labels=labels,use_cache=False,return_dict=True);nll=F.cross_entropy(output.logits[0].float(),labels[0],reduction='none')[lexical];return env['summarize_nll'](nll.cpu().numpy())
 proof=[];values=[]
 for i,r in enumerate(rows):
  p=out/'rows'/(r['signal']+'.npz')
  if p.exists():
   with np.load(p,allow_pickle=False)as z:assert str(z['audio_SHA'])==r['audio_SHA'];v=z['lm4'].copy();framegap=float(z['framegap']);decoder_gap=float(z['decoder_gap'])
  else:
   assert sha(r['audio'])==r['audio_SHA'];wave,sr=librosa.load(r['audio'],sr=16000);assert sr==16000;mel=processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt').input_features
   with torch.no_grad():h=encoder(mel.to(device)).last_hidden_state[0].cpu().numpy()
   assert h.shape==(1500,1280)and h.dtype==np.float32;v=score(h,r['prompt']);framegap=0.;decoder_gap=0.
   if i<2:
    old=np.load(pathlib.Path(cfg['controls'])/(r['signal']+'.npy'),allow_pickle=False);framegap=float(abs(h-old).max());decoder_gap=float(abs(score(old,r['prompt'])-ref[index[r['signal']]]).max())
   np.savez_compressed(p,signal=r['signal'],audio_SHA=r['audio_SHA'],lm4=v,framegap=framegap,decoder_gap=decoder_gap)
  if i<2:
   gap=float(abs(v-ref[index[r['signal']]]).max());proof.append(dict(signal=r['signal'],LM4_maxgap=gap,cached_frame_decoder_gap=decoder_gap,fresh_fullframe_gap=framegap));put('VALID_control_observations.json',dict(cases=proof,engineering_limit=cfg['maximum_lm4_gap']));assert gap<=cfg['maximum_lm4_gap']and decoder_gap<=cfg['maximum_lm4_gap'],proof
   if i==1:put('VALID_control_passed.json',dict(cases=proof))
  values.append(v)
  if i%16==0 or i==len(rows)-1:put('progress.json',dict(done=max(0,i-1),total=2189,seconds=time.monotonic()-start));print('LM4',max(0,i-1),flush=True)
 p=out/'EVAL_REFERENCE_LM4.npz';np.savez_compressed(p,signals=np.array([r['signal']for r in rows]),lm4=np.asarray(values));put('complete.json',dict(rows=2189,VALID_control_rows=2,output_SHA=sha(p),training_updates=0,VALID_controls=proof,seconds=time.monotonic()-start))

if __name__=='__main__':main()
