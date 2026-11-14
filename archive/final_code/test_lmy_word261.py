"""Frozen LMY paired 261-dimensional word evidence; VALID-gated EVAL inference."""
import argparse,ast,hashlib,json,math,re,sys,time,types,unicodedata
from pathlib import Path
from functools import lru_cache

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args()
 root=Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);(out/'rows').mkdir(exist_ok=True);start=time.monotonic()
 sys.path.insert(0,str(root/'tf_overlay'))
 import numpy as np,torch,soundfile as sf,librosa,transformers
 from torch import nn
 from torch.nn import functional as F
 from scipy.signal import resample_poly
 import transformers.utils.import_utils as availability
 availability._torchvision_available=False;availability._torchaudio_available=False
 availability.is_torchvision_available=lambda:False;availability.is_torchaudio_available=lambda:False
 transformers.utils.is_torchvision_available=lambda:False;transformers.utils.is_torchaudio_available=lambda:False
 from transformers import WhisperModel,WhisperFeatureExtractor
 assert torch.__version__=='2.6.0+cu124'and transformers.__version__=='5.14.1'
 torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,indent=2,allow_nan=False));p.replace(out/n)
 def sha(p):
  h=hashlib.sha256()
  with Path(p).open('rb')as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  return h.hexdigest()
 def require(ok,msg):
  if not ok:raise ValueError(msg)
 phones=tuple(('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH').split());vocab={n:i+1 for i,n in enumerate(phones)}
 def module(key,names=None,extra=None):
  p=root/'sources'/cfg['sources'][key]['name'];assert sha(p)==cfg['sources'][key]['SHA'];env=dict(np=np,torch=torch,nn=nn,F=F,math=math,re=re,unicodedata=unicodedata,Path=Path,lru_cache=lru_cache,require=require,sha=sha,FEATURE_DIM=261,PHONES=phones,VOCAB=vocab,DICT_HASH=cfg['dictionary_SHA'])
  if extra:env.update(extra)
  nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and(names is None or n.name in names)]
  if names is not None:assert {n.name for n in nodes}==set(names)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env);return types.SimpleNamespace(**env)
 core=module('core');ground=module('ground',['Matcher']);probe=module('probe',['lexicon','word_phones','blocks','text_arrays']);stream=module('stream',['query'],dict(normalized_words=core.normalized_words,scoring_mapping=core.scoring_mapping));words=module('words',['pronunciation','one']);fast=module('fast');labels=module('labels',['expanded_words'])
 lex=probe.lexicon(root/'cmudict.dict');contractions=cfg['contractions'];rows=json.loads((root/'selection.json').read_text());assert len(rows)==2191
 cp=torch.load(root/'true_pair_epoch3.pt',map_location='cpu',weights_only=True);assert cp['protocol_sha256']=='9228cc7314865684545ba077768c6915caea21374a3510d39ea6922c50c9765d'and cp['epoch']==3 and cp['arm']=='true_pair'
 matcher=ground.Matcher().to(device).eval().requires_grad_(False);matcher.load_state_dict(cp['model'])
 put('progress.json',dict(stage='loading_original_whisper',seconds=time.monotonic()-start))
 extractor=WhisperFeatureExtractor.from_pretrained(cfg['model'],local_files_only=True)
 full=WhisperModel.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa');encoder32=full.encoder.to(device).eval().requires_grad_(False);del full
 full=WhisperModel.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa');encoder16=full.encoder.to(device).eval().requires_grad_(False);del full
 with np.load(root/'valid_word_reference.npz',allow_pickle=False)as z:
  ids=z['signals'].tolist();off=np.r_[0,np.cumsum(z['lengths'])];gold={s:(z['features'][off[j]:off[j+1]].astype(np.float32),z['evidence_mask'][off[j]:off[j+1]].copy())for j,s in enumerate(ids[:2])}
 class Source:
  def audio1(self,sid):return self.heard,dict(source='original_librosa_mono_fp32_full1500_trim')
  def audio2(self,sid):return self.reference,dict(source='original_left_resample_poly_fp16_full1500_trim')
 source=Source();source.device=device;source.matchers={'true_pair':matcher};source.queries={};source.public={}
 proof=[];values=[];masks=[];lengths=[];pools=[]
 for i,row in enumerate(rows):
  sid=row['signal'];saved=out/'rows'/(sid+'.npz')
  if saved.exists():
   with np.load(saved,allow_pickle=False)as z:
    assert str(z['audio_SHA'])==row['audio_SHA']and str(z['audio2_SHA'])==row['audio2_SHA'];v=z['values'].copy();k=z['support'].copy();pooled=z['pooled'].copy();a1gap=float(z['audio1_control_gap'])
  else:
   assert sha(row['audio'])==row['audio_SHA']and sha(row['audio2'])==row['audio2_SHA']
   a1,sr=librosa.load(row['audio'],sr=16000);assert sr==16000
   mel=extractor(a1,sampling_rate=16000,return_tensors='pt').input_features
   with torch.no_grad():h=encoder32(mel.to(device)).last_hidden_state[0].cpu().numpy()
   assert h.shape==(1500,1280)and h.dtype==np.float32
   a1gap=0.
   if i<2:
    old=np.load(root/'controls'/(sid+'.npy'),allow_pickle=False);a1gap=float(abs(h-old).max());put('audio1_control_'+str(i)+'.json',dict(signal=sid,maxgap=a1gap,mean_gap=float(abs(h.mean(0)-old.mean(0)).max())))
    # The archived Source.audio1 contract gates the pooled coordinate at 1e-4.
    # Full-frame differences are recorded; the unchanged downstream word gate
    # below independently enforces the original fp16-spacing + 2e-5 tolerance.
    assert float(abs(h.mean(0)-old.mean(0)).max())<=1e-4,('Original pooled coordinate differs',sid,a1gap)
   pooled=h.mean(0);source.heard=h[:math.ceil(row['seconds']/.02)].copy()
   wave,sr=sf.read(row['audio2'],dtype='float32',always_2d=True);g=math.gcd(sr,16000);a2=resample_poly(wave[:,0],16000//g,sr//g).astype(np.float32)
   mel=extractor(a2,sampling_rate=16000,return_tensors='pt').input_features
   with torch.no_grad():h=encoder16(mel.to(device,dtype=torch.float16)).last_hidden_state
   source.reference=h[0,:math.ceil(row['seconds2']/.02)].float().cpu().numpy().copy();del h,mel
   public=dict(n_words=row['n_words'],words=labels.expanded_words(row['prompt'],contractions),files=dict(audio1=dict(seconds=row['seconds']),audio2=dict(seconds=row['seconds2'])))
   assert len(public['words'])==row['n_words'];source.public[sid]=public
   q=words.pronunciation(dict(original_prompt=row['original_prompt'],files=public['files']),public,probe,stream,lex,contractions);source.queries[sid]=q
   v,k,diag=words.one(source,sid,fast,core);put('query_'+sid+'.json',q)
   tmp=saved.with_suffix('.next.npz');np.savez_compressed(tmp,signal=sid,audio_SHA=row['audio_SHA'],audio2_SHA=row['audio2_SHA'],values=v,support=k,pooled=pooled,audio1_control_gap=a1gap);tmp.replace(saved)
  if i<2:
   old,keep=gold[sid];assert np.array_equal(k,keep);gap=abs(v.astype(np.float32)-old);tolerance=np.maximum(2e-5,np.spacing(abs(old).astype(np.float16)).astype(np.float32))+2e-5
   proof.append(dict(signal=sid,maxgap=float(gap.max()),audio1_maxgap=a1gap,exceeding_entries=int((gap>tolerance).sum())));put('VALID_control_observations.json',dict(cases=proof));assert(gap<=tolerance).all(),proof
   if i==1:put('VALID_control_passed.json',dict(cases=proof))
  else:values.append(v);masks.append(k);lengths.append(len(v));pools.append(pooled)
  if i%8==0 or i==len(rows)-1:put('progress.json',dict(done=max(0,i-1),total=2189,seconds=time.monotonic()-start));print('DONE',max(0,i-1),flush=True)
 assert all(torch.equal(v.detach().cpu(),cp['model'][n])for n,v in matcher.state_dict().items())
 p=out/'EVAL_WORD261_POOLED1280.npz';np.savez_compressed(p,signals=np.asarray([r['signal']for r in rows[2:]]),lengths=np.asarray(lengths),values=np.concatenate(values),support=np.concatenate(masks),pooled=np.asarray(pools));put('complete.json',dict(rows=2189,training_updates=0,output_SHA=sha(p),VALID_controls=proof,seconds=time.monotonic()-start))

if __name__=='__main__':main()
