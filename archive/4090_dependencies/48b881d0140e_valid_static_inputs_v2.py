"""Read-only static inputs for exact2117 VALID inference; no fitting or target APIs."""
import os,sys,json,hashlib,importlib.util,csv
from pathlib import Path
import numpy as np
import torch
if not hasattr(Path,'is_relative_to'):
 def _is_relative_to(self,other):
  try:self.relative_to(other);return True
  except ValueError:return False
 Path.is_relative_to=_is_relative_to
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OLD=A/'d6e_qwen_encoder_direct_swap_20261015';ORIGINAL=A/'d6e_listener_joint_seed44_valid_export_v1';BEST=A/'d6e_shared24_smallNN_beta_valid_export_side_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def save(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def prepare_static(out):
 out=Path(out);assert out.resolve().is_relative_to(ROOT.resolve())
 sys.path.insert(0,str(ORIGINAL))
 e=module('_qwen_valid_original_listener',ORIGINAL/'run.py');helper=e.own_exporter()
 swap=module('_qwen_valid_original_swap',OLD/'run_qwen_frame_swap_stage1.py');rf=swap.load_original_module();os.chdir(str(OLD))
 # Existing export guard blocks label/scoring/fitting entry points in own source.
 sys.setprofile(e.guard)
 ctx=rf.prepare(out,Path('/home/yangjing/workplace/clip2/clip2_next'))
 assert ctx['identity']==read(ROOT/'full_scale1521_v1/protocol.json')['public_input_identity']
 args=type('PublicArgs',(),{})();args.project=Path('/home/lisiyi/ICASSP2_2026');args.data=args.project/'data/clip2';args.peer=Path('/home/yangjing/workplace/clip2/clip2_next');args.valid_frames=args.project/'embeddings/whisper_large_v3_layer32_mono_fp32/valid';args.parent_valid=A/'d6e_pooled_seed43_valid_export_v1/results/export_seed43_v1'
 rr,ids,metadata=e.rows(args);byid={r['signal']:r for r in rr};index={s:i for i,s in enumerate(ids)}
 assert len(ids)==2117 and not set(ids)&set(ctx['frozen'].ids+ctx['frozen'].partition['excluded_train_ids'])
 assert sha(metadata)=='3197ba3ab089342e190ffd88773c20bf6ddf60d71f807714510cf08d8ae47a40'
 golden=BEST/'results/export_beta_raw_v1';gr=read(golden/'report.json')
 assert gr['checkpoint_sha256']=='cb8effb6c6e44a14f6cd6588528bd6b8ac0ac3750b5901438f6e5210551b5dd3'
 assert sha(golden/gr['csv']['file'])==gr['csv']['sha256']
 with np.load(golden/'official2117_raw_predictions.npz',allow_pickle=False) as z:
  assert z['signals'].tolist()==ids;baseline=z['raw'].astype(np.float64)
 with (golden/gr['csv']['file']).open(newline='') as f:csvrows=list(csv.reader(f))
 assert [r[0] for r in csvrows]==ids and len(csvrows)==2117 and all(len(r)==2 for r in csvrows)
 assert float(abs(np.array([float(r[1])/100 for r in csvrows])-baseline).max())<1e-11
 assert digest(ids)==gr['valid_ids_sha256']
 bankroot=ORIGINAL/'results/valid_listener_bank_v1';complete=read(bankroot/'complete.json');assert complete['n_valid']==2117 and complete['identity']['valid_metadata_sha256']==sha(metadata)
 listeners={}
 for receipt in complete['receipts']:
  p=bankroot/receipt['file'];assert Path(receipt['file']).name==receipt['file'] and sha(p)==receipt['sha256']
  part=e.chunk(p,receipt['signals'],byid,complete['identity']);assert not set(part)&set(listeners);listeners.update(part)
 assert set(listeners)==set(ids) and all(len(listeners[s])==byid[s]['n_words'] for s in ids)
 vp=read(args.parent_valid/'valid_protocol.json');assert vp['valid_metadata_sha256']==sha(metadata)
 h=ctx['old']['parent'];pa=h.parser().parse_args([]);pa.project,pa.data,pa.valid_frames=args.project,args.data,args.valid_frames
 oldplan=read(Path(ctx['frozen'].p['source'])/'results/clip2_full_v1/baseline_inputs.json');weights=read(Path(ctx['frozen'].p['readiness'])/'pretrained_weight_hashes.json')
 pooled,rawside,baseline_hash=h.dep.valid_baseline(pa,rr,oldplan['configuration'],weights['model_path']);assert baseline_hash==vp['baseline_input_sha256']
 side=ctx['retained']['infer'].normalized_side(dict(pooled=pooled,**rawside),ctx['retained']['bundle'].norm)
 evidence,word_bank=helper.valid_bank(rf,ids,rr,args.parent_valid,h.wf)
 text_asset=read(H/'MATCHED_TEXT_ASSET_V1.json');textpath=Path(text_asset['chosen_view'])
 tokens,vectors,masks=helper.text_features(rf,ctx,rr,args.peer,out,torch.device('cpu'),textpath)
 ti=read(out/'valid_text_identity.json');oldti=read(golden/'valid_text_identity.json')
 assert ti['new_public_keys']==0
 # Post-call backend JSON includes mutable truncation/padding serialization.
 # Do not accept this difference without the full original2117 golden replay in static_gate_v2.
 state_difference={k:{'current':ti.get(k),'prior':oldti.get(k)} for k in set(ti)|set(oldti) if ti.get(k)!=oldti.get(k)}
 assert set(state_difference)<= {'tokenizer_sha256'}
 assert sha(out/'valid_text_key_audit.json')==sha(golden/'valid_text_key_audit.json')
 actual_text_sha={name:hashlib.sha256(b''.join(p.encode()+b'\\0'+np.ascontiguousarray(value[p]).tobytes() for p in sorted(value))).hexdigest() for name,value in [('tokens',tokens),('word_vectors',vectors),('word_masks',masks)]}
 contract={'status':'static_original2117_inputs_prepared_no_Qwen_predictions','ids_sha256':digest(ids),'metadata_sha256':sha(metadata),'rows':2117,'maximum_words':max(r['n_words'] for r in rr),'public_input_identity_sha256':digest(ctx['identity']),'baseline_checkpoint_sha256':gr['checkpoint_sha256'],'baseline_CSV_sha256':gr['csv']['sha256'],'baseline_prediction_npz_sha256':sha(golden/'official2117_raw_predictions.npz'),'listener_complete_sha256':sha(bankroot/'complete.json'),'word_bank':word_bank,'baseline_raw_feature_files':baseline_hash,'text_identity':ti,'post_call_backend_state_difference':state_difference,'actual_text_arrays_sha256':actual_text_sha,'requires_original2117_golden_replay':True,'new_text_fitting_or_encoding':False,'normalization':'Existing TRAIN-only coordinates, independently checked maxdiff0','targets_selected':False,'fitting':False,'helper_source_sha256':sha(A/'d6e_boundary_seed44_valid_export_v1/run.py'),'listener_source_sha256':sha(ORIGINAL/'run.py')}
 save(out/'static_input_contract.json',contract)
 return dict(rf=rf,ctx=ctx,swap=swap,original=e,helper=helper,args=args,rows=rr,ids=ids,byid=byid,index=index,pooled=pooled,side=side,evidence=evidence,tokens=tokens,vectors=vectors,masks=masks,listeners=listeners,baseline=baseline,contract=contract)
