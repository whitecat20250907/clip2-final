"""Public references, teacher assignment and existing OOF lineage. No scoring targets."""
import sys,pathlib,json,hashlib,collections,datetime
sys.dont_write_bytecode=True
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;R=ROOT.parent;RUN=R.parent.parent
OOF=pathlib.Path('/root/autodl-tmp/clip2_runs/qwen3_listener_no_x_content_oof_20261015')
P=RUN/'training_pair_direct_v4/inputs'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,o):
 with (ROOT/n).open('x') as f:json.dump(o,f,indent=2,allow_nan=False)
plan=read(ROOT/'PREREGISTRATION_V1.json');clar=read(ROOT/'ARITHMETIC_AND_PUBLIC_PROMPT_CLARIFICATION_V1.json')
assert clar['correct_total_generation_rows']==6176 and plan['DEV_rows']+plan['VALID_rows']==6176
meta=read(P/'metadata.json');assert sha(P/'metadata.json')=='f2dbd62c9020f8ed8308aaa68f94501df285e079f3c535457870f7c3a51fbc07'
with np.load(P/'static.npz',allow_pickle=False) as z:
 prompt_index=z['prompt_index'].copy();wordmask=z['words'].copy()
fitids=read(RUN/'partition.json')['train_ids'];assert len(fitids)==16180 and digest(fitids)==plan['TRAIN_order_sha256']
predpath=OOF/'oof_predictions_v1/oof_predictions.jsonl';assert sha(predpath)=='480e84abaea40c3bd5d95e58f4b727b52dac692e7240b11cf0a0b87705aaf9fb'
pred=[json.loads(l) for l in predpath.read_text().splitlines() if l]
assert len(pred)==16204 and [r['signal'] for r in pred]==meta['ids'][:16204]
assert all(set(r)=={'signal','text','fold'} for r in pred)
idfold={r['signal']:int(r['fold']) for r in pred};assert set(idfold.values())=={0,1}
assert digest(meta['ids'][16204:])==plan['DEV_order_sha256']
public=[];promptfold={}
normalize_prompt=lambda x:' '.join(x.casefold().split())
for i,sid in enumerate(meta['ids']):
 ref=meta['prompts'][int(prompt_index[i])];n=int(wordmask[i].sum());assert len(ref.split())==n
 role='TRAIN_public' if i<16204 else 'DEV_public'
 row=dict(signal=sid,reference=ref,words=n,role=role)
 if i<16204:
  fold=idfold[sid];key=normalize_prompt(ref)
  if key in promptfold:assert promptfold[key]==fold
  else:promptfold[key]=fold
  row['fold']=fold
 public.append(row)
vpath=pathlib.Path('/root/autodl-tmp/clip2_data/cadenza_data/clip2/metadata/valid_metadata.json')
assert sha(vpath)=='3197ba3ab089342e190ffd88773c20bf6ddf60d71f807714510cf08d8ae47a40'
valid=read(vpath);assert len(valid)==2117 and all('correctness' not in r for r in valid)
for r in valid:
 assert len(r['prompt'].split())==r['n_words']
 public.append(dict(signal=r['signal'],reference=r['prompt'],words=int(r['n_words']),role='VALID_public'))
assert len(public)==22380 and len({r['signal'] for r in public})==22380
with np.load(R/'actual_parent_full22380_v1/frozen_parent_and_45features_all22380.npz',allow_pickle=False) as z:assert z['signals'].tolist()==[r['signal'] for r in public] and all(int(w.sum())==r['words'] for w,r in zip(z['words'],public))
ref_by_id={r['signal']:r['reference'] for r in public}
seen=0
for row in public[16204:]:
 key=normalize_prompt(row['reference'])
 if key in promptfold:row['fold']=promptfold[key];seen+=1;row['teacher_rule']='existing_TRAIN_prompt_heldout_fold'
 else:row['fold']=int(hashlib.sha256(('qwen-free-word-v1:'+key).encode()).hexdigest(),16)%2;row['teacher_rule']='unseen_prompt_fixed_hash'
teachers={};excluded=set(meta['ids'][:16204])-set(fitids);assert len(excluded)==24
for fold in [0,1]:
 config=read(OOF/('oof_predictions_v1/fold%d_config.json'%fold))
 cp=pathlib.Path(config['adapter_path']);expected=['c077c712f74c19429021a6db4dfa84d03a207e7936497b95ee6d2c9e369ba072','c23a6fc808496a4c7c3a5e2cb4401623f9fb4713f590190b234768e19a6e521c'][fold]
 assert cp.is_relative_to(OOF) and sha(cp/'adapter_model.safetensors')==expected
 manifest=OOF/('manifests/fold%d_train_no_x.jsonl'%fold);assert sha(manifest)==config['training_manifest_sha256']
 training_ids={json.loads(l)['signal'] for l in manifest.read_text().splitlines() if l}
 predicted_ids=[r['signal'] for r in pred if r['fold']==fold]
 assert not training_ids&set(predicted_ids)
 refs={normalize_prompt(ref_by_id[s]) for s in training_ids}
 assert not refs&{normalize_prompt(r['reference']) for r in public if r['fold']==fold}
 new_ids={r['signal'] for r in public[16204:] if r['fold']==fold};assert not new_ids&training_ids
 teachers[str(fold)]=dict(adapter_path=str(cp),adapter_sha256=expected,training_manifest_sha256=sha(manifest),TRAIN_source_rows=len(training_ids),current_strict24_ancestor_count=len(training_ids&excluded),TRAIN_prediction_rows=len(predicted_ids),new_generation_rows=len(new_ids),no_prediction_ID_or_canonical_prompt_in_teacher_training=True)
modelroot=pathlib.Path('/root/clip2_workspace/models/Qwen3-ASR-1.7B')
identitypaths=sorted(set(list(modelroot.glob('*.safetensors'))+[p for p in [modelroot/n for n in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','vocab.json','merges.txt','chat_template.json']] if p.exists()]))
modelidentity={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in identitypaths}
assert any(n.endswith('.safetensors') for n in modelidentity)
put('BASE_ASR_MODEL_INPUT_IDENTITY_V1.json',modelidentity)
generation=[]
for row in public[16204:]:
 role='valid' if row['role']=='VALID_public' else 'train'
 audio=pathlib.Path('/root/autodl-tmp/clip2_data/cadenza_data/clip2')/role/'signals'/(row['signal']+'.flac')
 assert audio.is_file()
 generation.append(dict(signal=row['signal'],audio=str(audio),fold=row['fold'],role=row['role']))
assert len(generation)==6176 and all(set(r)=={'signal','audio','fold','role'} for r in generation)
put('public_reference_records.json',public);put('generation_audio_records.json',generation)
put('teacher_sources.json',teachers)
put('PUBLIC_INPUT_AND_LINEAGE_GATE_V1.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='6176public_generation_and16204TRAIN_OOF_boundary_verified',records=22380,TRAIN_rows=16204,scorer_TRAIN_rows=16180,DEV_rows=4059,VALID_rows=2117,new_generation_rows=6176,existing_TRAIN_prompt_generations=seen,teacher_sources=teachers,prediction_sha256=sha(predpath),public_reference_sha256=sha(ROOT/'public_reference_records.json'),generation_records_sha256=sha(ROOT/'generation_audio_records.json'),teacher_sources_sha256=sha(ROOT/'teacher_sources.json'),base_model_identity_sha256=sha(ROOT/'BASE_ASR_MODEL_INPUT_IDENTITY_V1.json'),canonical_DEV_hash=digest(meta['ids'][16204:]),DEV_targets_read=False,human_responses_as_features=False,ASR_training_manifest_read_for_ID_boundary_only=True,wholepipelineOOF=False))
print(json.dumps(read(ROOT/'PUBLIC_INPUT_AND_LINEAGE_GATE_V1.json')),flush=True)
