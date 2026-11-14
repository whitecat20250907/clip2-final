"""Validate original4.57.6 frontend on64 retainedTRAIN, no encoder weights or labels."""
import os,sys,json,time,hashlib,importlib.util,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;OUT=H/'frontend64_v1';OUT.mkdir(exist_ok=False)
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','NUMBA_CACHE_DIR','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
with (OUT/'launch.json').open('x') as f:json.dump({'PID':os.getpid(),'training':False,'encoder_weights_loaded':False},f)
try:
 import numpy as np,torch,librosa
 torch.set_num_threads(2)
 from transformers import WhisperFeatureExtractor as Installed
 modern=module('_frontend_reference457',H/'whisper_frontend_4_57_6.py').WhisperFeatureExtractor
 cfg=read(H/'reference/preprocessor_config.json');old=Installed(**cfg);new=modern(**cfg);assert np.array_equal(old.mel_filters,new.mel_filters)
 selected=read(H/'real_audio64_v2/protocol.json')['ids']
 with np.load(ROOT/'qwen_first_shard_readonly_copy.npz',allow_pickle=False) as z:ids=z['signals'].tolist();ll=z['lengths']
 li={s:int(n) for s,n in zip(ids,ll)};rows=[];old_bad=0;maxmelgap=0
 data=Path('/home/lisiyi/ICASSP2_2026/data/clip2/train/signals')
 for j,sid in enumerate(selected):
  wav,_=librosa.load(data/(sid+'.flac'),sr=16000,mono=True)
  a=old([wav],sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='pt')
  b=new([wav],sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='pt')
  bad=int(a['attention_mask'].sum())!=a['input_features'].shape[-1];old_bad+=bad
  assert a['input_features'].shape==b['input_features'].shape
  gap=float(abs(a['input_features']-b['input_features']).max());maxmelgap=max(maxmelgap,gap);assert gap==0.
  ml=int(b['attention_mask'].sum());assert ml==b['input_features'].shape[-1]==len(wav)//160
  tail=ml%100;first=(tail-1)//2+1;pred=((first-1)//2+1-1)//2+1+(ml//100)*13
  assert pred==li[sid],(sid,pred,li[sid])
  rows.append({'signal':sid,'wave_samples':len(wav),'modern_mel_frames':ml,'old_mask_sum':int(a['attention_mask'].sum()),'expected_Qwen_frames':li[sid],'Mel_value_maxgap':gap})
  if (j+1)%8==0:print('FRONTEND_PROGRESS',j+1,64,'old_mask_mismatches',old_bad,flush=True)
 rep={'status':'reference457_frontend64_frames_match_cached_encoder_lengths','n':64,'old_mask_length_mismatches':old_bad,'modern_mask_length_mismatches':0,'max_Mel_value_gap_vs_installed':maxmelgap,'mel_filter_bank_identical':True,'reference_source_sha256':sha(H/'reference/feature_extraction_whisper_4_57_6.py'),'frontend_module_sha256':sha(H/'whisper_frontend_4_57_6.py'),'rows':rows,'training':False,'encoder_weights_loaded':False,'labels_read':False}
 with (OUT/'report.json').open('x') as f:json.dump(rep,f,indent=2)
 (OUT/'exitcode').write_text('0');print(json.dumps({k:v for k,v in rep.items() if k!='rows'}),flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
