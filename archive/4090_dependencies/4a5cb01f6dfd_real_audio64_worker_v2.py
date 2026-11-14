"""Real-audio parity gate for the frozen audio-only tower. No score/response labels."""
import os,sys,json,time,math,hashlib,datetime,importlib.util,traceback,subprocess
from pathlib import Path
from types import SimpleNamespace
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OLD=A/'d6e_qwen_encoder_direct_swap_20261015';OUT=H/'real_audio64_v2'
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
chosen_gpu=max(gpus)[2] if gpus else None
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
 from transformers import WhisperFeatureExtractor
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
 assert len(set(selected))==64
 data=Path('/home/lisiyi/ICASSP2_2026/data/clip2/train/signals');assert all((data/(s+'.flac')).is_file() for s in selected)
 prep=read(H/'reference/preprocessor_config.json');feature_extractor=WhisperFeatureExtractor(**prep)
 protocol={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'64 retainedTRAIN, deterministic length quantiles in pinned original256 shard, no target-based selection. Engineering parity only. No correctness or response reads.','ids':selected,'source_frame_sha256':sha(source),'audio_source':str(data),'audio_read_rule':'Exact original librosa.load(sr=16000,mono=True); native WhisperFeatureExtractor128, paddingTrue,truncationFalse,return_attention_maskTrue; trim mel to sum of mask; BF16 input','public_weights':'Pinned HF7278e1e70fe206f11671096ffdd38061171dd6e5 audio_tower397BF16 tensors,634955008bytes RAM only','public_audio_block_bytes':634955008,'normalizer':'Existing analytic RMS in frozen trained candidate; no fit','thresholds':{'max_clip_relative_feature_RMSE':.015,'min_clip_flattened_cosine':.9998,'max_final_probability_gap_points':.01,'RMS_final_probability_gap_points':.002},'runtime':{'torch':torch.__version__,'librosa':librosa.__version__,'numpy':np.__version__,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'GPU_cap_GiB':2},'attention':'Native0.0.6 copied forward and Torch SDPA; no mask/chunk changes. Wrapper forwards no explicit backend, original modern HF uses SDPA by default. Whole-real-frame parity remains required.','foundation_source_sha256':gate['source_reference_sha256'],'implementation_sha256':gate['implementation_sha256'],'script_sha256':sha(Path(__file__)),'training':False,'VALID_started':False}
 save('protocol.json',protocol)
 print('AUDIO_WEIGHT_STREAM_READY',634955008,flush=True)
 raw=bytearray(634955008);view=memoryview(raw);got=0
 while got<len(raw):
  n=sys.stdin.buffer.readinto(view[got:min(got+2**20,len(raw))]);assert n,'Short public weight stream';got+=n
 digest=sys.stdin.buffer.read(64).decode('ascii');assert len(digest)==64 and hashlib.sha256(raw).hexdigest()==digest
 assert sys.stdin.buffer.read(1)==b''
 save('public_weight_stream_receipt.json',{'bytes':len(raw),'SHA256':digest,'source':'Only public official audio-tower tensor range; not complete shard LFS hash verification','written_to_disk':False})
 device='cpu'
 if chosen_gpu is not None and torch.cuda.is_available():
  free,total=torch.cuda.mem_get_info(0)
  if free>=4*1024**3:
   torch.cuda.set_per_process_memory_fraction(2*1024**3/total,0);device='cuda:0'
 model=api.load_audio_block(cfg,header,raw,device=device);del raw,view
 records=[];newarrays=[]
 with torch.inference_mode():
  for j,sid in enumerate(selected):
   tick=time.monotonic();wav,_=librosa.load(data/(sid+'.flac'),sr=16000,mono=True)
   enc=feature_extractor([wav],sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='pt')
   ml=int(enc['attention_mask'][0].sum());mel=enc['input_features'][0,:,:ml].to(device=device,dtype=torch.bfloat16);lens=torch.tensor([ml],device=device,dtype=torch.long)
   hidden=model(mel,feature_lens=lens).last_hidden_state
   fresh=hidden.float().cpu().numpy().astype(np.float16);ii=qi[sid];n=int(ll[ii]);old=reference[int(offsets[ii]):int(offsets[ii])+n]
   assert fresh.shape==old.shape and fresh.shape[1]==2048 and np.isfinite(fresh).all()
   a=fresh.astype(np.float64).ravel();b=old.astype(np.float64).ravel();rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30));cos=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
   records.append({'signal':sid,'audio_sha256':sha(data/(sid+'.flac')),'wave_samples16k':len(wav),'mel_frames':ml,'encoder_frames':n,'relative_RMSE':rel,'flattened_cosine':cos,'maxabs_feature_gap':float(abs(a-b).max()),'seconds':time.monotonic()-tick})
   newarrays.append(fresh)
   if (j+1)%4==0:print('AUDIO_PARITY_PROGRESS',j+1,64,'maxrel',max(r['relative_RMSE'] for r in records),flush=True)
 save('feature_parity.json',{'records':records,'device':device,'peak_GPU_MiB':float(torch.cuda.max_memory_allocated()/2**20) if device.startswith('cuda') else 0.})
 feature_pass=max(r['relative_RMSE'] for r in records)<=.015 and min(r['flattened_cosine'] for r in records)>=.9998
 if not feature_pass:
  save('report.json',{'status':'feature_parity_failed_no_VALID','feature_pass':False,'max_relative_RMSE':max(r['relative_RMSE'] for r in records),'min_cosine':min(r['flattened_cosine'] for r in records),'seconds':time.monotonic()-started})
  (OUT/'exitcode').write_text('2');print('AUDIO_PARITY_FEATURE_GATE_FAILED',flush=True);sys.exit(2)
 # Frozen full-chain replay on the same64, without target or response labels.
 if device.startswith('cuda'):model.to('cpu');torch.cuda.empty_cache()
 del model
 swap=module('_audio_parity_swap',OLD/'run_qwen_frame_swap_stage1.py');rf=swap.load_original_module();os.chdir(str(OLD))
 ctx=rf.prepare(OUT,Path('/home/yangjing/workplace/clip2/clip2_next'));assert ctx['identity']==read(ROOT/'full_scale1521_v1/protocol.json')['public_input_identity']
 oldtext=ctx['frames'].text
 def stream(arrays,which):
  lens=np.array([len(v) for v in arrays],np.int64);return SimpleNamespace(index={s:i for i,s in enumerate(which)},lengths=lens,offsets=np.r_[0,np.cumsum(lens[:-1])],frames=np.concatenate(arrays),text=oldtext)
 oldarrays=[reference[int(offsets[qi[s]]):int(offsets[qi[s]])+int(ll[qi[s]])] for s in selected]
 m,control=swap.make_model(rf,ctx,torch.device('cpu'));del control
 cp=ROOT/'full_scale1521_v1/rms_qwen/seed44_boundary_joint_final.pt';assert sha(cp)==read(ROOT/'full_scale1521_v1/rms_qwen/parent_cache_report.json')['checkpoint_sha256'];m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['weights']);m.eval()
 stages={}
 with torch.inference_mode():
  for name,arr in [('reference',oldarrays),('reproduced',newarrays)]:
   ctx['frames']=stream(arr,selected);pp=[];zz=[]
   for k in range(0,64,16):
    ss=selected[k:k+16];x=list(swap.batch(ctx,ss,torch.device('cpu'),'boundary_joint'));x[5]=x[5]*torch.rsqrt(x[5].square().mean(-1,keepdim=True)+1e-6)*x[6][...,None]
    p,z=m(*x,mode='boundary_joint');pad=np.zeros((len(ss),12),np.float32);pad[:,:z.shape[1]]=z.numpy();pp.extend(p.tolist());zz.append(pad)
   stages[name]=(np.array(pp,np.float32),np.concatenate(zz))
 with np.load(commonpath,allow_pickle=False) as z:allids=z['signals'].tolist();common=tuple(z[k].copy() for k in ['evidence','support','words'])
 with np.load(ROOT/'full_scale1521_v1/rms_qwen/parent_logits_and_predictions.npz',allow_pickle=False) as z:assert z['signals'].tolist()==allids;cached=(z['logits'].copy(),common[0],common[1],common[2],z['prediction'].copy())
 index={s:i for i,s in enumerate(allids)};ix=np.array([index[s] for s in selected]);assert max(abs(stages['reference'][0]-cached[-1][ix]))<2e-6
 ds=module('_parity_downstream',OLD/'train_qwen_downstream.py');sys.path.insert(0,str(ds.LISTENER));f=module('_parity_listener',ds.LISTENER/'run_full_listener.py')
 Head=module('_parity_head',A/'d6e_boundary_listener_normalized_context_v1/tf_readout.py').TFReadout;CTC=module('_parity_ctc',A/'d6e_expert_bottleneck_diagnosis_v1/ctc_competition_readout.py').CTCReadout;Fine=module('_parity_fine',ds.BEST/'fine_residual.py').FineResidual
 b,a,donors,width,br=f.training_bank(ctx,OUT);b,a=f.complete_listener_bank(ctx,OUT,b,a,br,donors);di=np.array([ctx['index'][donors[s]] for s in ctx['frozen'].ids]);full=f.tensors(ctx,cached,b,a,di,ix,torch.device('cpu'),'adapted_listener')
 norm=read(ROOT/'full_readout_v1/rms_qwen/TRAIN_extra_normalization.json');extra=torch.from_numpy(np.clip((full[1][:,:,:24].numpy()-np.array(norm['mean']))/np.array(norm['std']),-8,8).astype(np.float32))
 point=Fine(CTC(Head(),104440),124443,'bounded_pooled');bc=ROOT/'full_readout_v1/rms_qwen/beta/rms_qwen_beta_final.pt';assert sha(bc)=='16a74314760abdfc2f59ced0f3306aa39af9e7c9a2a67313570457e68e0e4be2';point.load_state_dict(torch.load(bc,map_location='cpu',weights_only=True)['weights']);point.eval()
 final={}
 with torch.inference_mode():
  for name,(_,z) in stages.items():
   xx=list(full);xx[0]=torch.from_numpy(z);final[name]=point(*xx,extra)[0].numpy()
 delta=100*(final['reproduced'].astype(np.float64)-final['reference']);rms=float(np.sqrt(np.mean(delta*delta)));maxgap=float(abs(delta).max());passed=rms<=.002 and maxgap<=.01
 with (OUT/'frozen_score_parity.npz').open('xb') as f0:np.savez_compressed(f0,signals=np.array(selected),reference=final['reference'],reproduced=final['reproduced'])
 save('report.json',{'status':'real_audio_and_frozen_full_chain_parity_passed' if passed else 'score_parity_failed_no_VALID','n':64,'feature_pass':True,'max_relative_feature_RMSE':max(r['relative_RMSE'] for r in records),'min_cosine':min(r['flattened_cosine'] for r in records),'frozen_final_prediction_RMS_gap_points':rms,'frozen_final_prediction_maxgap_points':maxgap,'score_predictions_sha256':sha(OUT/'frozen_score_parity.npz'),'device':device,'training':False,'labels_read':False,'VALID_predictions_made':False,'seconds':time.monotonic()-started})
 (OUT/'exitcode').write_text('0' if passed else '2');print('AUDIO_PARITY_COMPLETE',passed,rms,maxgap,flush=True)
except SystemExit:raise
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
