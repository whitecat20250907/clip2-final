"""Original2117 static-input and formal-best32 replay gate. No new VALID candidate predictions."""
import os,sys,json,hashlib,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OUT=H/'valid_static_gate_v2'
OUT.mkdir(exist_ok=False)
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',POOLED_TEXT_RAM='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True;sys.path.insert(0,str(H))
def save(name,x):
 with (OUT/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
started=time.monotonic();save('launch.json',dict(pid=os.getpid(),training=False,candidate_VALID_predictions=False))
try:
 import numpy as np,torch
 torch.set_num_threads(2)
 import valid_static_inputs_v2 as st
 d=st.prepare_static(OUT);print('STATIC2117_PREPARED',flush=True)
 import valid_candidate_v1 as api
 # Same reviewed architecture; replace every parameter strictly with original formal-best.
 c=api.FrozenQwenCandidate('rms_qwen','cpu')
 cp=A/'d6e_shared24_smallNN_beta_valid_export_side_v1/results/fullTRAIN_joint_v1/beta_binomial_final.pt'
 assert st.sha(cp)=='cb8effb6c6e44a14f6cd6588528bd6b8ac0ac3750b5901438f6e5210551b5dd3'
 ck=torch.load(cp,map_location='cpu',weights_only=True)
 c.model.load_state_dict(ck['weights'],strict=True);c.model.eval().requires_grad_(False)
 p=A/'d6e_shared24_smallNN_beta_valid_export_side_v1/results/export_beta_raw_v1/public_replay_inputs32.npz'
 before=d['rf'].u.state_hash(c.model.state_dict())
 with np.load(p,allow_pickle=False) as z:
  x=tuple(torch.from_numpy(z['input_'+str(i)]) for i in range(7));expected=z['prediction'].copy()
  with torch.inference_mode():q,zlog=c.model(*x)
  got=q.numpy();wordgap=float(abs((zlog.sigmoid()*x[4]).sum(1)/x[4].sum(1)-q).max())
 gap=float(abs(got-expected).max());assert gap<2e-6 and wordgap<2e-6
 assert before==d['rf'].u.state_hash(c.model.state_dict())

 # Exact original Whisper frame parent with current static text mappings.
 rr,pr,oldreport,seal,parent_ck,cal=d['helper'].saved_candidate()
 assert pr['source_inputs']==d['ctx']['identity']
 parent=d['rf'].model_for(d['ctx'],44,torch.device('cpu')).eval().requires_grad_(False)
 parent.load_state_dict(parent_ck['weights'],strict=True)
 parent_before=d['rf'].u.state_hash(parent.state_dict())
 frames=d['helper'].cached_frames.ReviewedValidFrames(d['rf'],d['ctx'],d['ids'],d['args'].peer/'output/wavlm/features_whisper_last_valid')
 allq=[];first_fields=None
 for start in range(0,2117,32):
  take=d['ids'][start:start+32]
  xx=d['helper'].valid_batch(take,d['index'],d['pooled'],d['side'],d['evidence'],lambda sid:frames.get(sid,d['args'].data/'valid/signals'/(sid+'.flac')).astype(np.float32),d['tokens'],d['vectors'],d['masks'],d['byid'],torch.device('cpu'))
  listener=np.zeros(tuple(xx[4].shape)+(3,),np.float32)
  for j,sid in enumerate(take):listener[j,:len(d['listeners'][sid])]=d['listeners'][sid]
  with torch.inference_mode():_,logits=parent(*xx,mode='boundary_joint')
  l,e,li,support,words,severity=[logits.numpy(),xx[2].numpy(),listener,xx[3].numpy(),xx[4].numpy(),xx[-1].numpy()]
  api.validate_batch(l,e,li,support,words,severity)
  ev=api.normalize(e,c.context_norm['mean'],c.context_norm['std'],support)
  li=api.normalize(li,c.listener_norm['mean'],c.listener_norm['std'])
  extra=np.clip((ev[...,:24]-np.array(c.extra_norm['mean']))/np.array(c.extra_norm['std']),-8,8).astype(np.float32)
  fields=(l.astype(np.float32),ev,li,support,words,severity.astype(np.int64),extra)
  if start==0:first_fields=fields
  with torch.inference_mode():q,zlog=c.model(*(torch.from_numpy(v) for v in fields))
  wordgap=max(wordgap,float(abs((zlog.sigmoid()*xx[4]).sum(1)/xx[4].sum(1)-q).max()))
  allq.extend(q.tolist())
  if start%256==0:print('ORIGINAL_GOLDEN2117_REPLAY',start+len(take),flush=True)
 frames.verify_unchanged();frames.close();d['ctx']['frames'].verify_stat()
 assert parent_before==d['rf'].u.state_hash(parent.state_dict())
 assert before==d['rf'].u.state_hash(c.model.state_dict())
 fullgap=float(abs(np.array(allq)-d['baseline']).max());assert fullgap<2e-6 and wordgap<2e-6
 fieldgaps={}
 with np.load(p,allow_pickle=False) as z:
  for j,v in enumerate(first_fields):
   old=z['input_'+str(j)];assert v.shape==old.shape
   delta=v.astype(np.float64)-old.astype(np.float64)
   if j==0:delta=delta[first_fields[4].astype(bool)]
   fieldgaps[str(j)]=float(abs(delta).max());assert fieldgaps[str(j)]<2e-5
 with (OUT/'original2117_golden_replay.npz').open('xb') as f:np.savez_compressed(f,signals=np.array(d['ids']),expected=d['baseline'],got=np.array(allq))

 save('report.json',dict(status='original2117_static_inputs_and_formal_best32_replay_passed',n_valid=2117,n_replay=32,baseline_maxgap=gap,original2117_golden_replay_maxgap=fullgap,first32_actual_inputs_maxgaps=fieldgaps,original2117_golden_replay_sha256=st.sha(OUT/'original2117_golden_replay.npz'),text_state_hash_difference_adjudication='All static content identities exact; actual first32 head inputs and all2117 formalbest predictions replay below original tolerances. No numeric thresholds changed.',baseline_max_word_mean_gap=wordgap,baseline_checkpoint_sha256=st.sha(cp),baseline_replay_input_sha256=st.sha(p),static_contract_sha256=st.sha(OUT/'static_input_contract.json'),static_module_sha256=st.sha(H/'valid_static_inputs_v2.py'),seconds=time.monotonic()-started,training=False,labels_read=False,candidate_VALID_predictions=False))
 (OUT/'exitcode').write_text('0');print('VALID_STATIC_AND_BEST_GATE_PASSED',gap,fullgap,flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
