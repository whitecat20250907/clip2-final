"""Frozen RMS full-chain score parity for externally extracted fixed64 TRAIN frames.
Runs in original Torch2.4 scoring runtime. No targets, responses, fitting or VALID.
"""
import os,sys,json,time,hashlib,datetime,importlib.util,traceback
from pathlib import Path
if not hasattr(Path,'is_relative_to'):
 def relative(self,other):
  try:self.relative_to(other);return True
  except ValueError:return False
 Path.is_relative_to=relative
Q=Path(__file__).resolve().parent;H=Q.parent;A=H.parent;OLD=A/'d6e_qwen_encoder_direct_swap_20261015'
OUT=Q/'frozen_score64_torch291_v1';FEATURES=Q/'numeric_audio64_torch291_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def save(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
# Do not create a candidate output directory until full64 features passed unchanged gates.
f=read(FEATURES/'report.json');records=f['presets']['bf16_auto']
assert (FEATURES/'exitcode').read_text().strip()=='0' and f['n']==64 and len(records)==64
assert max(r['relative_RMSE'] for r in records)<=.015 and min(r['cosine'] for r in records)>=.9998
OUT.mkdir(exist_ok=False)
sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',RAYON_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',POOLED_TEXT_RAM='0')
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
save('launch.json',dict(PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=False,labels_read=False,VALID=False))
started=time.monotonic()
try:
 import numpy as np,torch
 assert torch.__version__.startswith('2.4.1');torch.set_num_threads(2)
 replay_root=Q/'numeric_fullchain64_v1';contract=read(replay_root/'public_replay_contract.json');rp=replay_root/contract['file']
 assert sha(rp)==contract['sha256']=='fdd3fd9a6149f2a28aed6c28836810e09e64da2d2ea708bfc42e50b086528a0b'
 with np.load(rp,allow_pickle=False) as z:r={k:z[k].copy() for k in z.files}
 ids=r['signals'].tolist();assert len(ids)==64 and ids==read(FEATURES/'protocol.json')['ids']
 with np.load(FEATURES/'fresh64_frames.npz',allow_pickle=False) as z:
  assert z['signals'].tolist()==ids
  lengths=z['lengths'].astype(np.int64);fresh=z['frames'].copy()
 assert lengths.shape==(64,) and np.all(lengths>0) and fresh.shape==(int(lengths.sum()),2048) and np.isfinite(fresh).all()
 fp=H/'qwen_first_shard_readonly_copy.npz';assert sha(fp)==contract['reference_frame_sha256']
 with np.load(fp,allow_pickle=False) as z:
  qi={s:i for i,s in enumerate(z['signals'].tolist())};ll=z['lengths'].astype(np.int64);offsets=np.r_[0,np.cumsum(ll[:-1])];old=z['frames'].copy()
 assert all(int(ll[qi[s]])==int(lengths[i]) for i,s in enumerate(ids))
 # Load original public constructor dependencies; scorer guard rejects target/fitting APIs.
 parent_export=A/'d6e_listener_joint_seed44_valid_export_v1'
 sys.path.insert(0,str(parent_export));guard=module('_score64_exportguard',parent_export/'run.py')
 swap=module('_score64_swap',OLD/'run_qwen_frame_swap_stage1.py');rf=swap.load_original_module();os.chdir(str(OLD));sys.setprofile(guard.guard)
 ctx=rf.prepare(OUT,Path('/home/yangjing/workplace/clip2/clip2_next'))
 assert ctx['identity']==read(H/'full_scale1521_v1/protocol.json')['public_input_identity']
 parent,control=swap.make_model(rf,ctx,torch.device('cpu'));del control
 cp=H/'full_scale1521_v1/rms_qwen/seed44_boundary_joint_final.pt';assert sha(cp)=='ecd30fd3f6e030353501049fc4159aa8b6fbd377c16d6e429d8ac028625791e3'
 parent.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['weights'],strict=True);parent.eval().requires_grad_(False)
 scoring=module('_score64_candidate',Q/'valid_candidate_v1.py');head=scoring.FrozenQwenCandidate('rms_qwen')
 before=[rf.u.state_hash(parent.state_dict()),rf.u.state_hash(head.model.state_dict())]
 padded_new=np.zeros((64,int(lengths.max()),2048),np.float32);padded_ref=np.zeros_like(padded_new);pos=0
 for i,s in enumerate(ids):
  n=int(lengths[i]);padded_new[i,:n]=fresh[pos:pos+n];pos+=n
  oi=qi[s];padded_ref[i,:n]=old[offsets[oi]:offsets[oi]+n]
 mask=r['raw_input_6'];assert mask.shape==padded_new.shape[:2] and np.array_equal(mask,np.arange(int(lengths.max()))[None,:]<lengths[:,None])
 result={}
 with torch.inference_mode():
  for name,frames in [('reference',padded_ref),('reproduced',padded_new)]:
   x=[torch.from_numpy(frames) if i==5 else torch.from_numpy(r['raw_input_'+str(i)]) for i in range(12)]
   x[5]=x[5]*torch.rsqrt(x[5].square().mean(-1,keepdim=True)+1e-6)*x[6][...,None]
   _,logits=parent(*x,mode='boundary_joint')
   xx=[logits]+[torch.from_numpy(r['head_input_'+str(i)]) for i in range(1,6)]+[torch.from_numpy(r['head_extra'])]
   q,word_logits=head.model(*xx);result[name]=q.numpy()
   if name=='reference':
    legal=r['head_input_4'];gap=float(abs(logits.numpy()[legal]-r['parent_logits_reference'][legal]).max())
    assert gap<2e-5 and float(abs(result[name]-r['final_reference']).max())<2e-6
 assert before==[rf.u.state_hash(parent.state_dict()),rf.u.state_hash(head.model.state_dict())]
 delta=100*(result['reproduced'].astype(np.float64)-r['final_reference'])
 rms=float(np.sqrt(np.mean(delta**2)));maximum=float(abs(delta).max());passed=rms<=.002 and maximum<=.01
 with (OUT/'frozen_score_parity.npz').open('xb') as ff:np.savez_compressed(ff,signals=np.array(ids),reference=r['final_reference'],reference_replay=result['reference'],reproduced=result['reproduced'])
 save('report.json',dict(status='full64_features_and_frozen_score_parity_passed' if passed else 'full64_score_parity_failed_no_VALID',n=64,feature_max_relative_RMSE=max(x['relative_RMSE'] for x in records),feature_min_cosine=min(x['cosine'] for x in records),score_RMS_gap_points=rms,score_max_gap_points=maximum,score_pass=passed,reference_legal_logits_gap=gap,thresholds_unchanged=True,feature_report_sha256=sha(FEATURES/'report.json'),fresh_frames_sha256=sha(FEATURES/'fresh64_frames.npz'),replay_sha256=sha(rp),prediction_npz_sha256=sha(OUT/'frozen_score_parity.npz'),parent_checkpoint_sha256=sha(cp),head_checkpoint_sha256=sha(head.checkpoint),model_states_unchanged=True,labels_read=False,training=False,VALID=False,seconds=time.monotonic()-started))
 (OUT/'exitcode').write_text('0' if passed else '2')
 print('FROZEN_SCORE64_COMPLETE',passed,rms,maximum,flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
finally:sys.setprofile(None)
