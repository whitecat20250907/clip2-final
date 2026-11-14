
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common_v1 import *
z=nz(R/'LOCAL_DEPENDENCE_PUBLIC5120_BEFORE_NEW_TARGET_FITS_V1.npz',keys=['signals','features','legal','diagnostics'])
assert z['signals'].tolist()==mids
f=z['features'][:nf][z['legal'][:nf]];d=z['diagnostics'][:nf][z['legal'][:nf]]
def q(v):return np.quantile(v,[0,.1,.25,.5,.75,.9,1]).tolist()
out=dict(role='Public_features_only_original4096FIT_legalwords_no_targets',rows=4096,words=len(f),original_global_probability_quantiles=q(f[:,0]),original_local_probability_quantiles=q(f[:,3]),original_global_below_logit_floor_fraction=float((f[:,0]<=1e-8).mean()),original_local_below_logit_floor_fraction=float((f[:,3]<=1e-8).mean()),target_global_probability_delta_quantiles=q(f[:,6]),target_local_probability_delta_quantiles=q(f[:,9]),wrong_global_probability_delta_quantiles=q(f[:,12]),wrong_local_probability_delta_quantiles=q(f[:,15]),target_global_logit_delta_quantiles=q(f[:,7]),target_local_logit_delta_quantiles=q(f[:,10]),public_occupancy_mass_covered_quantiles=q(d[:,4]),energy_matching_log_distance_quantiles=q(d[:,0]),exact_target_global_zero_delta_fraction=float((f[:,6]==0).mean()),exact_target_local_zero_delta_fraction=float((f[:,9]==0).mean()),target_global_probability_decrease_fraction=float((f[:,6]<0).mean()),target_local_probability_decrease_fraction=float((f[:,9]<0).mean()),interpretation='Public measurement diagnostics only; not human causal effects, correctness labels or performance evidence.',new_fits=False,no_human_targets_read=True)
put('LOCAL_DEPENDENCE_FIT_PUBLIC_MEASUREMENT_DYNAMIC_RANGE_DIAGNOSTIC_V1.json',out);note(stage='local_dependence_public_dynamic_range',**out)
