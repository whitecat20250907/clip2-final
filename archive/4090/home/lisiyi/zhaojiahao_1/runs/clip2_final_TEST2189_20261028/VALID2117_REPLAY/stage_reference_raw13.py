from pathlib import Path
import json,hashlib,numpy as np
r=Path('/home/lisiyi/zhaojiahao_1/runs/clip2_final_TEST2189_20261028');out=r/'VALID2117_REPLAY';H=Path('/home/lisiyi/clip2_lmy/CLIP2_lmy/my_experiments/d6e_qwen_internalization_autoresearch_20261015');F=H/'strict24_native_decoder_evidence_v1/autodl_native_feature_recovery_v2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mp=F/'manifest.json';assert sha(mp)=='cda70bf941acd2bd0fe5be94da94e757914a3773ccc6474ede696a94621dbbc7';m=json.loads(mp.read_text());rows=json.loads((out/'selection.json').read_text());ids=[x['signal']for x in rows];wanted=set(ids);data={};proof={}
for e in m['chunks']:
 if e['stop']<=20263:continue
 p=F/e['file'];assert sha(p)==e['sha256'];proof[str(p)]=e['sha256']
 with np.load(p,allow_pickle=False)as z:
  ss=z['signals'].tolist();features=z['word_features'];audio=z['audio_sha256'].tolist()
  for j,s in enumerate(ss):
   if s in wanted:data[s]=(features[j,:,:,1].copy(),audio[j])
assert set(data)==wanted
for row in rows:assert data[row['signal']][1]==row['audio_SHA']
p=out/'ORIGINAL_VALID_RAW13.npz';np.savez_compressed(p,signals=np.asarray(ids),raw13=np.stack([data[s][0]for s in ids]))
cfg=json.loads((out/'parent_protocol.json').read_text());cfg['assets']['reference_raw13']={'path':str(p),'SHA':sha(p)};orig=json.loads((r/'LMY_FINAL_READOUT/original_manifest.json').read_text())
for seed in [44,45]:
 p=out/('seed%d_sealed_Qwen_RMS_human_encoder_strict24_VALID2117.npz'%seed);cfg['assets']['human_reference'+str(seed)]={'path':str(p),'SHA':sha(p)};cfg['assets']['main_reference'+str(seed)]=orig['assets']['main_sealed'+str(seed)]
(out/'parent_protocol.json').write_text(json.dumps(cfg,indent=2));(out/'original_raw13_qualification.json').write_text(json.dumps(dict(rows=2117,all_original_audio_SHA_match=True,manifest_SHA=sha(mp),chunk_SHA=proof,labels_read=False),indent=2));print('Original VALID raw13 and static references verified',flush=True)
