"""Original2117 static-input and formal-best32 replay gate. No new VALID candidate predictions."""
import os,sys,json,hashlib,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OUT=H/'valid_static_gate_v1'
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
 import valid_static_inputs_v1 as st
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
 save('report.json',dict(status='original2117_static_inputs_and_formal_best32_replay_passed',n_valid=2117,n_replay=32,baseline_maxgap=gap,baseline_max_word_mean_gap=wordgap,baseline_checkpoint_sha256=st.sha(cp),baseline_replay_input_sha256=st.sha(p),static_contract_sha256=st.sha(OUT/'static_input_contract.json'),static_module_sha256=st.sha(H/'valid_static_inputs_v1.py'),seconds=time.monotonic()-started,training=False,labels_read=False,candidate_VALID_predictions=False))
 (OUT/'exitcode').write_text('0');print('VALID_STATIC_AND_BEST_GATE_PASSED',gap,flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
