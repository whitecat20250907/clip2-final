"""RMS Qwen frozen original2117 scoring with original AutoDL frames; no fitting/targets."""
import os,sys,time,json,hashlib,traceback,io,struct,datetime,csv
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OUT=H/'autodl_frozen_VALID2117_v1'
OUT.mkdir(exist_ok=False);sys.dont_write_bytecode=True;sys.path.insert(0,str(H))
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',POOLED_TEXT_RAM='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def exact(n):
 out=bytearray()
 while len(out)<n:
  v=sys.stdin.buffer.read(min(n-len(out),2**20))
  if not v:raise EOFError('Incomplete public feature stream')
  out.extend(v)
 return bytes(out)
started=time.monotonic()
save('launch.json',dict(PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Frozen RMS-only candidate export using originalAutoDLencoder; no retraining, no fitting, no targets'))
try:
 import numpy as np,torch
 torch.set_num_threads(2)
 import valid_static_inputs_v2 as st,valid_candidate_v1 as scoring
 assert read(H/'valid_static_gate_v3/report.json')['status']=='original2117_static_inputs_and_formal_best32_replay_passed'
 assert read(H/'static2117_independent_v1/report.json')['status']=='independent_original2117_golden_prediction_array_verification_passed'
 assert read(H/'scoring_port_gate_v3/independent_array_verification_v1.json')['pass_all']
 d=st.prepare_static(OUT)
 assert sha(OUT/'static_input_contract.json')==read(H/'valid_static_gate_v3/report.json')['static_contract_sha256']
 print('AUTODL_EXPORT_STATIC_READY',flush=True)
 parent,control=d['swap'].make_model(d['rf'],d['ctx'],torch.device('cpu'));del control
 cp=ROOT/'full_scale1521_v1/rms_qwen/seed44_boundary_joint_final.pt'
 assert sha(cp)=='ecd30fd3f6e030353501049fc4159aa8b6fbd377c16d6e429d8ac028625791e3'
 parent.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['weights'],strict=True)
 parent.eval().requires_grad_(False);head=scoring.FrozenQwenCandidate('rms_qwen','cpu')
 before=[d['rf'].u.state_hash(parent.state_dict()),d['rf'].u.state_hash(head.model.state_dict())]
 proof=H/'autodl_native64_inputs_v1';gate=read(proof/'report.json')
 assert gate['passed'] and gate['n']==64 and gate['thresholds_unchanged']
 fp=proof/'fresh64_frames.npz';assert sha(fp)==gate['frames_npz_sha256']
 rp=H/'numeric_fullchain64_v1/public_fullchain_replay64.npz'
 assert sha(rp)==read(H/'numeric_fullchain64_v1/public_replay_contract.json')['sha256']
 with np.load(rp,allow_pickle=False) as z:r={k:z[k].copy() for k in z.files}
 with np.load(fp,allow_pickle=False) as z:
  assert z['signals'].tolist()==r['signals'].tolist()
  lengths=z['lengths'].astype(np.int64);offset=np.r_[0,np.cumsum(lengths[:-1])];ff=z['frames'].copy()
 padded=np.zeros((64,int(lengths.max()),2048),np.float32)
 for j,n in enumerate(lengths):padded[j,:n]=ff[offset[j]:offset[j]+n]
 x=[torch.from_numpy(padded if i==5 else r['raw_input_'+str(i)]) for i in range(12)]
 assert np.array_equal(r['raw_input_6'],np.arange(padded.shape[1])[None,:]<lengths[:,None])
 x[5]=x[5]*torch.rsqrt(x[5].square().mean(-1,keepdim=True)+1e-6)*x[6][...,None]
 with torch.inference_mode():
  _,z=parent(*x,mode='boundary_joint')
  q=head.model(*([z]+[torch.from_numpy(r['head_input_'+str(i)]) for i in range(1,6)]+[torch.from_numpy(r['head_extra'])]))[0].numpy()
 delta=100*(q.astype(np.float64)-r['final_reference'])
 rms=float(np.sqrt(np.mean(delta**2)));maxgap=float(abs(delta).max())
 passed=rms<=.002 and maxgap<=.01
 gate64=dict(passed=passed,feature_gate_passed=gate['passed'],score_gate_passed=passed,n=64,final_RMS_gap_points=rms,final_maxgap_points=maxgap,feature_max_relative_RMSE=gate['max_relative_RMSE'],feature_min_cosine=gate['min_cosine'],feature_source_report_sha256=sha(proof/'report.json'),frozen_parent_sha256=sha(cp),frozen_head_sha256=sha(head.checkpoint),thresholds_unchanged=True,labels_read=False)
 save('fullchain64_pass.json',gate64);assert passed,'Fixed64fullchain drift gate failed; no VALID'
 print('AUTODL_FULLCHAIN64_PASSED',rms,maxgap,flush=True)
 del x,r,ff,padded,q
 ids=d['ids'];pred=[];maximum=max(row['n_words'] for row in d['rows'])
 wp=np.zeros((2117,maximum),np.float32);wordmask=np.arange(maximum)[None,:]<np.array([r['n_words'] for r in d['rows']])[:,None]
 receipts=[]
 print('READY_FOR_VALID_FEATURE_STREAM',flush=True)
 for start in range(0,2117,32):
  size=struct.unpack('!Q',exact(8))[0];assert 0<size<64*2**20
  raw=exact(size);digest=hashlib.sha256(raw).hexdigest()
  with np.load(io.BytesIO(raw),allow_pickle=False) as z:
   take=z['signals'].tolist();assert take==ids[start:start+32]
   lengths=z['lengths'].astype(np.int64);frames=z['frames'].copy()
  assert frames.dtype==np.float16 and frames.shape==(int(lengths.sum()),2048) and np.isfinite(frames).all()
  offset=np.r_[0,np.cumsum(lengths[:-1])]
  audio={s:frames[offset[j]:offset[j]+lengths[j]].astype(np.float32) for j,s in enumerate(take)}
  xx=scoring.valid_batch(take,d['index'],d['pooled'],d['side'],d['evidence'],lambda s:audio[s],d['tokens'],d['vectors'],d['masks'],d['byid'],torch.device('cpu'))
  listener=np.zeros(tuple(xx[4].shape)+(3,),np.float32)
  for j,s in enumerate(take):
   assert len(d['listeners'][s])==int(xx[4][j].sum())
   listener[j,:len(d['listeners'][s])]=d['listeners'][s]
  qq=head.score(parent,xx,listener);pred.extend(qq.astype(float).tolist())
  for j,s in enumerate(take):
   n=d['byid'][s]['n_words'];wp[start+j,:n]=head.last_word_probabilities[j,:n]
  receipts.append(dict(start=start,count=len(take),bytes=size,sha256=digest))
  print('RMS_VALID_SCORED',start+len(take),2117,'seconds',round(time.monotonic()-started,1),flush=True)
 assert before==[d['rf'].u.state_hash(parent.state_dict()),d['rf'].u.state_hash(head.model.state_dict())]
 d['ctx']['frames'].verify_stat()
 q=np.array(pred,np.float64);assert q.shape==(2117,) and np.isfinite(q).all() and 0<=q.min()<=q.max()<=1
 gap=float(abs((wp*wordmask).sum(1)/wordmask.sum(1)-q).max());assert gap<2e-6
 csvfile=OUT/'CLIP2_valid_Qwen_RMS_frozen_seed44_20261015.csv'
 with csvfile.open('x',newline='') as f:
  w=csv.writer(f)
  for sid,v in zip(ids,q):w.writerow([sid,format(100*float(v),'.10f')])
 with csvfile.open(newline='') as f:rows=list(csv.reader(f))
 assert len(rows)==2117 and [v[0] for v in rows]==ids and all(len(v)==2 for v in rows)
 roundtrip=float(abs(np.array([float(v[1])/100 for v in rows])-q).max());assert roundtrip<=1e-11
 p=OUT/'sealed_Qwen_RMS_VALID2117.npz'
 with p.open('xb') as f:np.savez_compressed(f,signals=np.array(ids),q=q,word_probabilities=wp,wordmask=wordmask,baseline_whisper=d['baseline'])
 report=dict(status='frozen_RMS_Qwen_original2117_export_complete_unsubmitted',n=2117,csv_file=csvfile.name,csv_sha256=sha(csvfile),ids_sha256=st.digest(ids),predictions_npz_sha256=sha(p),fullchain64_report_sha256=sha(OUT/'fullchain64_pass.json'),static_contract_sha256=sha(OUT/'static_input_contract.json'),parent_checkpoint_sha256=sha(cp),head_checkpoint_sha256=sha(head.checkpoint),model_states_unchanged=True,word_mean_maxgap=gap,csv_roundtrip_maxgap_0to1=roundtrip,minimum_score=float(q.min()*100),maximum_score=float(q.max()*100),feature_shards=receipts,training=False,labels_read=False,calibration=False,blend=False,official_submission=False,best_replaced=False,seconds=time.monotonic()-started)
 save('report.json',report);(OUT/'exitcode').write_text('0')
 print('RMS_QWEN_EXPORT_COMPLETE',json.dumps(report),flush=True)
except BaseException:
 sys.setprofile(None);traceback.print_exc();(OUT/'exitcode').write_text('1');raise
finally:sys.setprofile(None)
