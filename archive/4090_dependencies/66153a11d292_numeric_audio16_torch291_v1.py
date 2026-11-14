"""Real-audio parity gate for the frozen audio-only tower. No score/response labels."""
import os,sys,json,time,math,hashlib,datetime,importlib.util,traceback,subprocess
from pathlib import Path
from types import SimpleNamespace
if not hasattr(Path,'is_relative_to'):
 def _path_is_relative_to(self,other):
  try:self.relative_to(other);return True
  except ValueError:return False
 Path.is_relative_to=_path_is_relative_to
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OLD=A/'d6e_qwen_encoder_direct_swap_20261015';OUT=H/'numeric_audio16_torch291_v1'
OUT.mkdir(exist_ok=False)
log=(OUT/'run.log').open('x',buffering=1)
class Tee:
 def __init__(self,other):self.other=other
 def write(self,s):self.other.write(s);log.write(s);return len(s)
 def flush(self):self.other.flush();log.flush()
sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr);sys.dont_write_bytecode=True
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',RAYON_NUM_THREADS='2',POOLED_TEXT_RAM='0')
# Use at most2GiB on an idle GPU with >=8GiB free; otherwise CPU. Never touch other jobs.
gpus=[]
try:
 rr=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 for row in rr.stdout.splitlines():
  i,free,util=map(int,row.split(','))
  if free>=8192 and util<=5:gpus.append((free,-i,i))
except Exception:pass
chosen_gpu=(None if os.environ['CLIP_FIXED_GPU']=='cpu' else int(os.environ['CLIP_FIXED_GPU'])) if 'CLIP_FIXED_GPU' in os.environ else (max(gpus)[2] if gpus else None)
os.environ['CUDA_VISIBLE_DEVICES']='' if chosen_gpu is None else str(chosen_gpu)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 hh=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):hh.update(b)
 return hh.hexdigest()
def save(name,obj):
 with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
started=time.monotonic()
save('launch.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_idle_candidate':chosen_gpu,'GPU_allocator_cap_GiB':2,'training':False})
try:
 import numpy as np,torch,librosa
 frontend=module('_real_frontend457',H/'whisper_frontend_4_57_6.py')
 WhisperFeatureExtractor=frontend.WhisperFeatureExtractor
 frontend_gate=read(H/'frontend64_v1/report.json');assert frontend_gate['status']=='reference457_frontend64_frames_match_cached_encoder_lengths' and sha(H/'whisper_frontend_4_57_6.py')==frontend_gate['frontend_module_sha256']
 torch.set_num_threads(2)
 gate=read(H/'structural_preflight_v1/report.json');assert gate['status']=='audio_only_structure_and_attention_math_passed'
 assert sha(H/'audio_only.py')==gate['implementation_sha256']
 api=module('_real_audio_only',H/'audio_only.py')
 cfg=read(ROOT/'native_decoder_feasibility_v1/public_metadata_relay_v1/config.json')['thinker_config']['audio_config']
 header=read(ROOT/'audio_tower_weight_range_preflight_v1/model-00001-of-00002.safetensors.header.json')
 source=ROOT/'qwen_first_shard_readonly_copy.npz';assert sha(source)=='1ccc7928bf73928bda921299f58fba81bd101b70f78667a74f66283f7e940cb5'
 with np.load(source,allow_pickle=False) as z:
  ids=z['signals'].tolist();ll=z['lengths'].astype(np.int64);reference=z['frames'].copy()
 offsets=np.r_[0,np.cumsum(ll[:-1])];qi={s:i for i,s in enumerate(ids)}
 commonpath=OLD/'qwen_parent_word_cache_sharded_v1/qwen_parent_word_cache.npz'
 with np.load(commonpath,allow_pickle=False) as z:trainset=set(z['signals'].tolist()[:16204])
 eligible=sorted([s for s in ids if s in trainset],key=lambda s:(int(ll[qi[s]]),s));assert len(eligible)>=64
 selected=[eligible[int(j)] for j in np.rint(np.linspace(0,len(eligible)-1,64)).astype(int)]
 selected=[selected[int(j)] for j in np.rint(np.linspace(0,63,16)).astype(int)]
 assert len(set(selected))==16
 data=Path('/home/lisiyi/ICASSP2_2026/data/clip2/train/signals');assert all((data/(s+'.flac')).is_file() for s in selected)
 prep=read(H/'reference/preprocessor_config.json');feature_extractor=WhisperFeatureExtractor(**prep)
 protocol={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'64 retainedTRAIN, deterministic length quantiles in pinned original256 shard, no target-based selection. Engineering parity only. No correctness or response reads.','ids':selected,'source_frame_sha256':sha(source),'audio_source':str(data),'audio_read_rule':'Exact original librosa.load(sr=16000,mono=True); native WhisperFeatureExtractor128, paddingTrue,truncationFalse,return_attention_maskTrue; trim mel to sum of mask; BF16 input','public_weights':'Pinned HF7278e1e70fe206f11671096ffdd38061171dd6e5 audio_tower397BF16 tensors,634955008bytes RAM only','public_audio_block_bytes':634955008,'normalizer':'Existing analytic RMS in frozen trained candidate; no fit','thresholds':{'max_clip_relative_feature_RMSE':.015,'min_clip_flattened_cosine':.9998,'max_final_probability_gap_points':.01,'RMS_final_probability_gap_points':.002},'runtime':{'frontend_reference':'Transformers4.57.6 verbatim class, installed4.46 stable base imports','frontend_module_sha256':sha(H/'whisper_frontend_4_57_6.py'),'torch':torch.__version__,'librosa':librosa.__version__,'numpy':np.__version__,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'GPU_cap_GiB':2},'attention':'Native0.0.6 copied forward and Torch SDPA; no mask/chunk changes. Wrapper forwards no explicit backend, original modern HF uses SDPA by default. Whole-real-frame parity remains required.','foundation_source_sha256':gate['source_reference_sha256'],'implementation_sha256':gate['implementation_sha256'],'script_sha256':sha(Path(__file__)),'training':False,'VALID_started':False,'diagnostic':'Reference-feature-only backend/precision diagnosis, not an accuracy experiment'}
 save('protocol.json',protocol)
 ms=read(ROOT/'public_modelscope_header_preflight_v1/report.json');assert ms['status']=='official_ModelScope_header_matches_pinned_HF'
 print('PUBLIC_WEIGHT_DIRECT_DOWNLOAD_BEGIN',634955008,flush=True)
 import urllib.request
 raw=bytearray(634955008);got=0;attempt=0;mark=128*2**20
 while got<len(raw):
  attempt+=1;assert attempt<=5,'Public direct weight retries exhausted'
  try:
   req=urllib.request.Request(ms['public_source'],headers={'Range':'bytes=%d-635035191'%(80184+got),'User-Agent':'CLIP2-public-audio-only-parity'})
   with urllib.request.urlopen(req,timeout=40) as response:
    assert response.status==206 and response.headers.get('Content-Range')=='bytes %d-635035191/4220320824'%(80184+got)
    while got<len(raw):
     chunk=response.read(min(2**20,len(raw)-got))
     if not chunk:raise IOError('Incomplete public tensor range')
     raw[got:got+len(chunk)]=chunk;got+=len(chunk)
     if got>=mark:print('PUBLIC_DIRECT_PROGRESS',got,len(raw),flush=True);mark+=128*2**20
  except Exception as ex:
   print('PUBLIC_DIRECT_RETRY',attempt,got,type(ex).__name__,str(ex)[:200],flush=True);time.sleep(2)
 digest=hashlib.sha256(raw).hexdigest();assert digest=='d0906056e3f1a1de086aab7d6eba13b59e3719c73738dbff803dbc4a931c6021'
 print('PUBLIC_DIRECT_WEIGHT_SHA_VERIFIED',digest,flush=True)
 save('public_weight_stream_receipt.json',{'bytes':len(raw),'SHA256':digest,'source':'Official Qwen ModelScope tensor range with identical pinned HF header and previously verified audio-blockSHA; not complete shard LFS hash verification','written_to_disk':False})
 device='cpu'
 if chosen_gpu is not None and torch.cuda.is_available():
  free,total=torch.cuda.mem_get_info(0)
  if free>=4*1024**3:
   torch.cuda.set_per_process_memory_fraction(2*1024**3/total,0);device='cuda:0'
 model=api.load_audio_block(cfg,header,raw,device=device);del raw

 from contextlib import nullcontext
 presets=['bf16_auto']
 original_attn=model.config._attn_implementation
 all_records={};timings={}
 for preset in presets:
  tick=time.monotonic();model.config._attn_implementation='eager' if preset=='bf16_eager' else 'sdpa'
  dtype=torch.float32 if preset=='fp32_math' else torch.bfloat16
  model.to(dtype=dtype)
  records=[];fresh_frames=[]
  try:
   for j,sid in enumerate(selected):
    wav,_=librosa.load(data/(sid+'.flac'),sr=16000,mono=True)
    enc=feature_extractor([wav],sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='pt')
    ml=int(enc['attention_mask'][0].sum());mel=enc['input_features'][0,:,:ml].to(device=device,dtype=dtype)
    context=nullcontext()
    if device.startswith('cuda') and preset in ['bf16_flash','bf16_math','fp32_math']:
     context=torch.backends.cuda.sdp_kernel(enable_flash=preset=='bf16_flash',enable_math=preset!='bf16_flash',enable_mem_efficient=False)
    with context,torch.inference_mode():fresh=model(mel,feature_lens=torch.tensor([ml],device=device,dtype=torch.long)).last_hidden_state.float().cpu().numpy().astype(np.float16)
    fresh_frames.append(fresh)
    ii=qi[sid];old=reference[int(offsets[ii]):int(offsets[ii])+int(ll[ii])];assert fresh.shape==old.shape
    aa=fresh.astype(np.float64).ravel();bb=old.astype(np.float64).ravel()
    records.append(dict(signal=sid,relative_RMSE=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30)),cosine=float(np.dot(aa,bb)/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-30)),maxabs_gap=float(abs(aa-bb).max())))
   with (OUT/'fresh16_frames.npz').open('xb') as ff:np.savez_compressed(ff,signals=np.array(selected),lengths=np.array([len(x) for x in fresh_frames],np.int64),frames=np.concatenate(fresh_frames))
   all_records[preset]=records;print('NUMERIC_DIAG',preset,'maxrel',max(r['relative_RMSE'] for r in records),'mincos',min(r['cosine'] for r in records),flush=True)
  except Exception as ex:
   all_records[preset]={'error':type(ex).__name__,'message':str(ex)[:2000]};print('NUMERIC_DIAG_ERROR',preset,repr(ex),flush=True)
  timings[preset]=time.monotonic()-tick
 save('report.json',dict(status='numeric_reference_feature_diagnostic_complete_no_VALID',device=device,n=16,presets=all_records,seconds_by_preset=timings,training=False,labels_read=False,thresholds_changed=False,VALID_predictions=False))
 (OUT/'exitcode').write_text('0');print('NUMERIC_DIAG_COMPLETE',flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
