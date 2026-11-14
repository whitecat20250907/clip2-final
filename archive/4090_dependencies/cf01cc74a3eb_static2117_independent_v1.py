"""Independent saved-array verification of original formal-best2117 replay. No model imports."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;A=H.parent.parent;G=H/'valid_static_gate_v3';OUT=H/'static2117_independent_v1';OUT.mkdir(exist_ok=False)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 import numpy as np
 for i in range(180):
  if (G/'exitcode').exists():break
  try:os.kill(2912604,0)
  except ProcessLookupError:raise RuntimeError('Golden replay missing without exit marker; do not infer pass')
  time.sleep(10)
 assert (G/'exitcode').read_text().strip()=='0'
 r=read(G/'report.json');assert r['status']=='original2117_static_inputs_and_formal_best32_replay_passed'
 p=G/'original2117_golden_replay.npz';assert sha(p)==r['original2117_golden_replay_sha256']
 c=read(G/'static_input_contract.json');assert sha(G/'static_input_contract.json')==r['static_contract_sha256']
 old=A/'d6e_shared24_smallNN_beta_valid_export_side_v1/results/export_beta_raw_v1/official2117_raw_predictions.npz'
 assert sha(old)==c['baseline_prediction_npz_sha256']
 with np.load(p,allow_pickle=False) as z,np.load(old,allow_pickle=False) as b:
  assert z['signals'].tolist()==b['signals'].tolist();ids=z['signals'].tolist()
  assert len(ids)==len(set(ids))==2117
  assert z['expected'].shape==z['got'].shape==b['raw'].shape==(2117,)
  assert np.array_equal(z['expected'],b['raw'])
  gap=float(abs(z['got'].astype(np.float64)-b['raw'].astype(np.float64)).max())
  assert gap<2e-6 and np.isfinite(z['got']).all() and z['got'].min()>=0 and z['got'].max()<=1
 assert gap==r['original2117_golden_replay_maxgap']
 assert r['baseline_maxgap']<2e-6 and r['baseline_max_word_mean_gap']<2e-6
 assert all(v<2e-5 for v in r['first32_actual_inputs_maxgaps'].values())
 assert r['baseline_checkpoint_sha256']=='cb8effb6c6e44a14f6cd6588528bd6b8ac0ac3750b5901438f6e5210551b5dd3'
 prior=read(A/'d6e_shared24_smallNN_beta_valid_export_side_v1/results/export_beta_raw_v1/valid_text_identity.json')
 now=c['text_identity']
 assert {k:v for k,v in prior.items() if k!='tokenizer_sha256'}=={k:v for k,v in now.items() if k!='tokenizer_sha256'}
 assert now['new_public_keys']==0 and now['listener_responses_used']==False and now['model_fitting']==False
 assert c['requires_original2117_golden_replay'] and c['post_call_backend_state_difference']
 out=dict(status='independent_original2117_golden_prediction_array_verification_passed',n=2117,complete_original_ID_order=True,baseline_expected_exact=True,actual_maxgap01=gap,reference_prediction_sha256=sha(old),replayed_array_sha256=sha(p),gate_report_sha256=sha(G/'report.json'),static_contract_sha256=sha(G/'static_input_contract.json'),scope='Confirms stored2117 replay against immutable earlier formalbest outputs; first32 actualinput check inspected from gate report/code. Does not certify Qwen feature fidelity.',Qwen_VALID_allowed=False,labels_scored=False)
 with (OUT/'report.json').open('x') as f:json.dump(out,f,indent=2)
 (OUT/'exitcode').write_text('0');print('INDEPENDENT_ORIGINAL2117_VERIFIED',json.dumps(out),flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
