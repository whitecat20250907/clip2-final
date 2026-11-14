
from common_v1 import *
from collections import defaultdict,Counter
source=R.parent/'full_support_CTC_word_expectation_micro_v1/frozen_word_posterior_fresh_group_confirmation_v1/FROZEN_MODELS_FRESH_GROUP_ROLES_BEFORE_TARGETS_V1.npz'
old=nz(source);oldids=old['signals'].tolist();assert len(oldids)==1026 and set(oldids)<=set(train)
def group(s):return assets[s]['files']['audio2']['pcm_sha256']
microgroups={group(s)for s in mids};oldgroups={group(s)for s in oldids};devgroups={group(s)for s in dev}
bygroup=defaultdict(list)
for s in train:
 g=group(s)
 if g not in microgroups and g not in oldgroups and g not in devgroups:bygroup[g].append(s)
ordered=sorted(bygroup,key=lambda g:hashlib.sha256(('local_dependence_prospective_confirmation_v1|'+g).encode()).hexdigest())
chosen=[];gs=[]
for g in ordered:
 gs.append(g);chosen.extend(bygroup[g])
 if len(chosen)>=1024:break
assert chosen,'No available additional group'
assert set(chosen)<=set(train)and not set(chosen)&set(mids+oldids+dev)
chosen_groups={group(s)for s in chosen}
assert not chosen_groups&(microgroups|oldgroups|devgroups)
for g in chosen_groups:assert set(bygroup[g])=={s for s in chosen if group(s)==g}
chosen=[s for s in train if s in set(chosen)]
# Keep original TRAIN order after public group selection; no target fields are read.
save('PROSPECTIVE_TRAIN_GROUP_CONFIRMATION_ROLES_BEFORE_PRIMARY_SCORES_V1.npz',signals=np.array(chosen),groups=np.array([group(s)for s in chosen]))
put('PROSPECTIVE_TRAIN_GROUP_CONFIRMATION_SELECTION_BEFORE_PRIMARY_SCORES_V1.json',dict(timestamp=time.time(),role='Stage-new conditionalTRAINconfirmation only; NOTnewDEV/untouchedtest/wholepipelineOOF',selection='Audio2PCM groups disjointcurrent5120micro,prior1026confirmation,andoriginalDEV; deterministicSHAgrouporder,wholeTRAINgroupsuntilatleast1024rows,retainoriginalTRAINorder',available_groups=len(bygroup),available_rows=sum(len(v)for v in bygroup.values()),rows=len(chosen),groups=len(chosen_groups),severity_counts=dict(Counter(str(assets[s]['severity'])for s in chosen)),signals_sha=hashlib.sha256(json.dumps(chosen,separators=(',',':')).encode()).hexdigest(),rolefile_sha=sha(R/'PROSPECTIVE_TRAIN_GROUP_CONFIRMATION_ROLES_BEFORE_PRIMARY_SCORES_V1.npz'),old_confirmation_roles_sha=sha(source),strictTRAIN_ID_sha=bp['TRAIN_ID_SHA'],formalDEV_ID_sha=bp['DEV_ID_SHA'],source_public_asset_sha=sha(bp['word_public']),no_targets_or_humanresponses_read=True,no_models_or_audio_inference=True,only_use_if_primary_candidate_promising=True,original4059_membership_unchanged=True,fullstrict16180_training_membership_unchanged=True,legacy_upstream_ancestry_still_notOOF=True))
note(stage='prospective_conditional_TRAIN_confirmation_roles_only',rows=len(chosen),groups=len(chosen_groups),available_groups=len(bygroup))
