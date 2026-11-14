"""Frozen paired raw/RMS2117 VALID export. Only run after immutable real-audio and scoring gates."""
import os,sys,json,time,hashlib,datetime,importlib.util,traceback,subprocess,csv
from pathlib import Path
if not hasattr(Path,'is_relative_to'):
 def _is_relative_to(self,other):
  try:self.relative_to(other);return True
  except ValueError:return False
 Path.is_relative_to=_is_relative_to
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OLD=A/'d6e_qwen_encoder_direct_swap_20261015';OUT=H/'frozen_VALID2117_v1'
OUT.mkdir(exist_ok=False)
sys.dont_write_bytecode=True;sys.path.insert(0,str(H))
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',RAYON_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',POOLED_TEXT_RAM='0')
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
try:
 fidelity=H/'real_audio64_v6';port=H/'scoring_port_gate_v3';staticgate=H/'valid_static_gate_v1'
 for d in [fidelity,port,staticgate]:assert (d/'exitcode').read_text().strip()=='0',str(d)
 assert read(fidelity/'report.json')['status']=='real_audio_and_frozen_full_chain_parity_passed'
 assert read(port/'independent_array_verification_v1.json')['pass_all']
 assert read(staticgate/'report.json')['status']=='original2117_static_inputs_and_formal_best32_replay_passed'
 gpus=[]
 try:
  rr=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  for row in rr.stdout.splitlines():
   i,free,util=map(int,row.split(','))
   if free>=8192 and util<=5:gpus.append((free,-i,i))
 except Exception:pass
 chosen_gpu=max(gpus)[2] if gpus else None
 os.environ['CUDA_VISIBLE_DEVICES']='' if chosen_gpu is None else str(chosen_gpu)
 save('launch.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_idle_candidate':chosen_gpu,'GPU_allocator_cap_GiB':2,'training':False})
 import numpy as np,torch,librosa
 torch.set_num_threads(2)
 import valid_static_inputs_v1 as st,valid_candidate_v1 as scoring
 d=st.prepare_static(OUT);rf=d['rf'];ctx=d['ctx'];swap=d['swap']
 assert st.sha(OUT/'static_input_contract.json')==read(staticgate/'report.json')['static_contract_sha256']
 # Exactly frozen parent/head pairs; inference stays on CPU for the paired scoring-port identity.
 parents={};heads={};before={}
 for arm in ['raw_qwen','rms_qwen']:
  parent,control=swap.make_model(rf,ctx,torch.device('cpu'));del control
  cp=ROOT/'full_scale1521_v1'/arm/'seed44_boundary_joint_final.pt'
  assert sha(cp)==read(ROOT/'full_scale1521_v1'/arm/'parent_cache_report.json')['checkpoint_sha256']
  parent.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['weights'],strict=True);parent.eval().requires_grad_(False)
  parents[arm]=parent;heads[arm]=scoring.FrozenQwenCandidate(arm,'cpu')
  before[arm]=[rf.u.state_hash(parent.state_dict()),rf.u.state_hash(heads[arm].model.state_dict())]
 protocol={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'No-fit paired raw/RMS Qwen Audio1 frame replacement, exact2117 public VALID. No calibration, no blend, no weight selection, no target reads.','output':'100 times legal-word sigmoid mean; CSV no header two columns, IDs unchanged','static_contract_sha256':sha(OUT/'static_input_contract.json'),'gates':{str(p):sha(p) for p in [fidelity/'report.json',port/'independent_array_verification_v1.json',staticgate/'report.json']},'code':{str(p):sha(p) for p in [Path(__file__),H/'valid_candidate_v1.py',H/'valid_static_inputs_v1.py',H/'audio_only.py',H/'whisper_frontend_4_57_6.py']},'weights':{a:{'parent':sha(ROOT/'full_scale1521_v1'/a/'seed44_boundary_joint_final.pt'),'head':sha(heads[a].checkpoint)} for a in parents},'runtime':{'torch':torch.__version__,'librosa':librosa.__version__,'numpy':np.__version__,'scorers_device':'cpu'},'train_steps':0,'labels_read':False,'official_submission':False,'best_replaced':False}
 save('protocol.json',protocol)
 api=module('_valid_audio_only',H/'audio_only.py');frontend=module('_valid_frontend457',H/'whisper_frontend_4_57_6.py')
 cfg=read(ROOT/'native_decoder_feasibility_v1/public_metadata_relay_v1/config.json')['thinker_config']['audio_config'];header=read(ROOT/'audio_tower_weight_range_preflight_v1/model-00001-of-00002.safetensors.header.json')
 feature_extractor=frontend.WhisperFeatureExtractor(**read(H/'reference/preprocessor_config.json'))
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

 def encode(sid,folder):
  p=folder/(sid+'.flac');fingerprint=sha(p)
  wav,_=librosa.load(p,sr=16000,mono=True)
  enc=feature_extractor([wav],sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='pt')
  ml=int(enc['attention_mask'][0].sum());assert ml==enc['input_features'].shape[-1]
  with torch.inference_mode():v=model(enc['input_features'][0,:,:ml].to(device=device,dtype=torch.bfloat16),feature_lens=torch.tensor([ml],device=device,dtype=torch.long)).last_hidden_state.float().cpu().numpy().astype(np.float16)
  assert v.ndim==2 and v.shape[1]==2048 and len(v)>0 and np.isfinite(v).all()
  return v,dict(signal=sid,audio_sha256=fingerprint,wave_samples16k=len(wav),mel_frames=ml,encoder_frames=len(v))
 # Recheck actual tower/device, same predeclared64 IDs, fixed thresholds before any candidate VALID.
 replay=read(fidelity/'public_replay_contract.json');rp=fidelity/replay['file'];assert sha(rp)==replay['sha256']
 fp=ROOT/'qwen_first_shard_readonly_copy.npz';assert sha(fp)==replay['reference_frame_sha256']
 with np.load(fp,allow_pickle=False) as z:
  qi={s:i for i,s in enumerate(z['signals'].tolist())};lengths=z['lengths'].astype(np.int64);offsets=np.r_[0,np.cumsum(lengths[:-1])];reference=z['frames'].copy()
 with np.load(rp,allow_pickle=False) as z:replay_arrays={k:z[k].copy() for k in z.files}
 selected=replay_arrays['signals'].tolist();assert selected==read(fidelity/'protocol.json')['ids']
 fresh=[];feature_records=[]
 with torch.inference_mode():
  for j,sid in enumerate(selected):
   v,rec=encode(sid,Path('/home/lisiyi/ICASSP2_2026/data/clip2/train/signals'));i=qi[sid];old=reference[offsets[i]:offsets[i]+lengths[i]]
   assert v.shape==old.shape
   aa=v.astype(np.float64).ravel();bb=old.astype(np.float64).ravel()
   rec.update(relative_RMSE=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30)),cosine=float(np.dot(aa,bb)/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-30)))
   fresh.append(v);feature_records.append(rec)
   if (j+1)%16==0:print('ACTUAL_DEVICE_AUDIO64',j+1,flush=True)
 assert max(r['relative_RMSE'] for r in feature_records)<=.015 and min(r['cosine'] for r in feature_records)>=.9998
 width=max(map(len,fresh));padded=np.zeros((64,width,2048),np.float32)
 for j,v in enumerate(fresh):padded[j,:len(v)]=v
 x=[torch.from_numpy(replay_arrays['raw_input_'+str(i)]) if i!=5 else torch.from_numpy(padded) for i in range(12)]
 assert x[6].shape==padded.shape[:2]
 x[5]=x[5]*torch.rsqrt(x[5].square().mean(-1,keepdim=True)+1e-6)*x[6][...,None]
 with torch.inference_mode():
  _,logits=parents['rms_qwen'](*x,mode='boundary_joint')
  hs=[logits]+[torch.from_numpy(replay_arrays['head_input_'+str(i)]) for i in range(1,6)]+[torch.from_numpy(replay_arrays['head_extra'])]
  got=heads['rms_qwen'].model(*hs)[0].numpy()
 delta=100*(got.astype(np.float64)-replay_arrays['final_reference']);rms=float(np.sqrt(np.mean(delta**2)));maxgap=float(abs(delta).max())
 save('actual_device64_gate.json',dict(device=device,feature_records=feature_records,final_RMS_gap_points=rms,final_maxgap_points=maxgap,passed=rms<=.002 and maxgap<=.01,labels_read=False))
 assert rms<=.002 and maxgap<=.01,'Actual device final chain mismatch; no VALID'
 del reference,replay_arrays,fresh,padded,x,hs
 print('ACTUAL_DEVICE_GATE_PASSED',rms,maxgap,flush=True)
 ids=d['ids'];predictions={arm:[] for arm in parents};maxwords=max(r['n_words'] for r in d['rows'])
 wp={arm:np.zeros((2117,maxwords),np.float32) for arm in parents};wordmask=np.arange(maxwords)[None,:]<np.array([r['n_words'] for r in d['rows']])[:,None];audio_records=[]
 for start in range(0,2117,32):
  take=ids[start:start+32];audio={}
  for sid in take:
   v,rec=encode(sid,d['args'].data/'valid/signals');audio[sid]=v.astype(np.float32);audio_records.append(rec)
  x=scoring.valid_batch(take,d['index'],d['pooled'],d['side'],d['evidence'],lambda sid:audio[sid],d['tokens'],d['vectors'],d['masks'],d['byid'],torch.device('cpu'))
  listener=np.zeros(tuple(x[4].shape)+(3,),np.float32)
  for j,sid in enumerate(take):
   assert len(d['listeners'][sid])==int(x[4][j].sum())
   listener[j,:len(d['listeners'][sid])]=d['listeners'][sid]
  for arm in parents:
   q=heads[arm].score(parents[arm],x,listener);predictions[arm].extend(q.astype(float).tolist())
   for j,sid in enumerate(take):
    n=d['byid'][sid]['n_words'];wp[arm][start+j,:n]=heads[arm].last_word_probabilities[j,:n]
  print('FROZEN_VALID_PROGRESS',start+len(take),2117,'seconds',round(time.monotonic()-started,1),flush=True)
 del model
 if device.startswith('cuda'):torch.cuda.empty_cache()
 for arm in parents:
  assert before[arm]==[rf.u.state_hash(parents[arm].state_dict()),rf.u.state_hash(heads[arm].model.state_dict())]
 ctx['frames'].verify_stat();assert len(audio_records)==2117 and [r['signal'] for r in audio_records]==ids
 csvs={}
 for arm in parents:
  q=np.asarray(predictions[arm],np.float64);assert q.shape==(2117,) and np.isfinite(q).all() and q.min()>=0 and q.max()<=1
  assert float(abs((wp[arm]*wordmask).sum(1)/wordmask.sum(1)-q).max())<2e-6
  p=OUT/(arm+'_frozen_VALID2117.csv')
  with p.open('x',newline='') as f:
   writer=csv.writer(f)
   for sid,pred in zip(ids,q):writer.writerow([sid,format(float(pred)*100,'.10f')])
  with p.open(newline='') as f:rows=list(csv.reader(f))
  assert len(rows)==2117 and [r[0] for r in rows]==ids and all(len(r)==2 for r in rows)
  gap=float(abs(np.array([float(r[1])/100 for r in rows])-q).max());assert gap<=1e-11
  csvs[arm]=dict(file=p.name,sha256=sha(p),rows=2117,header=False,roundtrip_maxgap01=gap,min_score=float(q.min()*100),max_score=float(q.max()*100))
 with (OUT/'sealed_frozen_predictions2117.npz').open('xb') as f:
  np.savez_compressed(f,signals=np.asarray(ids),raw_qwen=np.array(predictions['raw_qwen'],np.float64),rms_qwen=np.array(predictions['rms_qwen'],np.float64),baseline_whisper=d['baseline'],wordmask=wordmask,raw_word_probs=wp['raw_qwen'],rms_word_probs=wp['rms_qwen'])
 save('audio_fingerprints.json',dict(records=audio_records,model_revision='7278e1e70fe206f11671096ffdd38061171dd6e5',audio_block_sha256='d0906056e3f1a1de086aab7d6eba13b59e3719c73738dbff803dbc4a931c6021',device=device))
 save('report.json',dict(status='frozen_raw_RMS_Qwen_original2117_VALID_export_complete_unsubmitted',n_valid=2117,protocol_sha256=sha(OUT/'protocol.json'),csvs=csvs,predictions_npz_sha256=sha(OUT/'sealed_frozen_predictions2117.npz'),valid_ids_sha256=st.digest(ids),audio_fingerprints_sha256=sha(OUT/'audio_fingerprints.json'),static_contract_sha256=sha(OUT/'static_input_contract.json'),actual_device64_gate_sha256=sha(OUT/'actual_device64_gate.json'),model_state_before_after_equal=True,training_steps=0,labels_read=False,VALID_fitting=False,calibration=False,blend=False,official_submission=False,best_replaced=False,seconds=time.monotonic()-started))
 (OUT/'exitcode').write_text('0');print('FROZEN_VALID_EXPORT_COMPLETE_UNSUBMITTED',json.dumps(csvs),flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
