
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common_v1 import *
records={}
for shard in range(3):
 d=read(R/('LOCAL_DEPENDENCE_FEATURE_SHARD%d_COMPLETE_V1.json'%shard))
 for rec in d['records']:
  lo=int(rec['file'].split('_')[-2]);assert (lo//64)%3==shard and lo not in records
  fp=R/'chunks'/rec['file'];assert sha(fp)==rec['sha'];z=nz(fp)
  assert z['signals'].tolist()==mids[lo:lo+64]and z['features'].shape==(len(mids[lo:lo+64]),12,26)
  expected=np.array([[w<len(assets[s]['words'])for w in range(12)]for s in mids[lo:lo+64]])
  assert np.array_equal(z['legal'],expected)
  assert z['audio1_sha'].tolist()==[assets[s]['files']['audio1']['sha256']for s in mids[lo:lo+64]]
  records[lo]=rec
assert sorted(records)==list(range(0,len(mids),64))
z=[nz(R/'chunks'/records[lo]['file'])for lo in sorted(records)]
features=np.concatenate([a['features']for a in z]);legal=np.concatenate([a['legal']for a in z]);diag=np.concatenate([a['diagnostics']for a in z])
assert np.isfinite(features).all()and np.isfinite(diag).all()and features.shape==(5120,12,26)
save('LOCAL_DEPENDENCE_PUBLIC5120_BEFORE_NEW_TARGET_FITS_V1.npz',signals=np.array(mids),features=features,legal=legal,diagnostics=diag)
put('LOCAL_DEPENDENCE_ALL_PUBLIC_FEATURES_SEALED_BEFORE_TARGET_FITS_V1.json',dict(rows=5120,legalwords=int(legal.sum()),feature_sha=sha(R/'LOCAL_DEPENDENCE_PUBLIC5120_BEFORE_NEW_TARGET_FITS_V1.npz'),records=[records[k]for k in sorted(records)],diagnostics=dict(max_RMS_gap=float(diag[...,3].max()),max_wrong_overlap=float(diag[...,2].max()),mean_energy_log_distance=float(diag[...,0][legal].mean()),mean_old_far_energy_log_distance=float(diag[...,1][legal].mean())),no_human_targets_or_responses=True,ASR_optimizer_steps=0,wholepipelineOOF=False))
note(stage='local_dependence_all5120_public_features_sealed')
