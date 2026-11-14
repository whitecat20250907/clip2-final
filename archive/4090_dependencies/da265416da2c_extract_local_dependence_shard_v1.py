
from local_dependence_asr_core_v1 import *
SHARD=int(os.environ['CLIP2_SHARD']);assert 0<=SHARD<3
start=time.monotonic();proof=[];first=[]
prior=R.parent/'local_word_energy_matched_intervention_CPU_feasibility_v1/bounded_local_span_native_ASR_feasibility_v2'
qs=read(prior/'BOUNDED_LOCAL_NATIVE_PUBLIC_ASR_FEASIBILITY_RESULT_V2.json');assert sha(prior/'BOUNDED_LOCAL_NATIVE_PUBLIC_ASR_FEASIBILITY_RESULT_V2.json')==PLAN['prior_native2FIT_sha']
oldfile=prior/'FIRST2FIT_PUBLIC_BOUNDED_LOCAL_NATIVE_FEATURES_V2.npz';assert sha(oldfile)==qs['feature_sha'];old=nz(oldfile)
for i in range(2):
 feat,ll,diag,rows=measure(i,proof=True);gap=float(abs(feat[ll,:12]-old['features'][i,ll,:12]).max());assert gap<1e-12
 proof.extend(rows);first.append((feat,ll,diag));note(stage='local_dependence_geometry_ASR_qualification',shard=SHARD,signal=mids[i],replayed_original_and_target_maxgap=gap,seconds=time.monotonic()-start)
put('LOCAL_DEPENDENCE_NATIVE_QUAL_SHARD%d_V1.json'%SHARD,dict(qualified=True,proof=proof,original_and_target_replay=True,energy_control_atleast_as_close_as_faredge=True,no_targets=True,ASR_updates=0,GPU_UUID=os.environ['CLIP2_GPU'],seconds=time.monotonic()-start))
chunks=R/'chunks';chunks.mkdir(exist_ok=True);records=[]
for lo in range(SHARD*64,len(mids),3*64):
 group=mids[lo:lo+64];fn='LOCAL_NATIVE_%04d_V1.npz'%lo;fp=chunks/fn;mf=fp.with_suffix('.json')
 if fp.exists()or mf.exists():
  assert fp.exists()and mf.exists();rec=read(mf);assert rec['signals']==group and sha(fp)==rec['sha'];records.append(rec);continue
 ff=[];ll=[];dd=[]
 for i in range(lo,min(lo+64,len(mids))):
  feat,legal,diag=(first[i]if i<2 else measure(i)[:3]);ff.append(feat);ll.append(legal);dd.append(diag)
 with fp.open('xb')as f:np.savez(f,signals=np.array(group),features=np.array(ff),legal=np.array(ll),diagnostics=np.array(dd),audio1_sha=np.array([assets[s]['files']['audio1']['sha256']for s in group]))
 rec=dict(file=fn,signals=group,rows=len(group),sha=sha(fp))
 with mf.open('x')as f:json.dump(rec,f,indent=2)
 records.append(rec);note(stage='local_dependence_public_feature_chunk',shard=SHARD,done=min(lo+64,len(mids)),total=len(mids),seconds=time.monotonic()-start)
put('LOCAL_DEPENDENCE_FEATURE_SHARD%d_COMPLETE_V1.json'%SHARD,dict(shard=SHARD,records=records,qualification_sha=sha(R/('LOCAL_DEPENDENCE_NATIVE_QUAL_SHARD%d_V1.json'%SHARD)),no_human_targets_or_responses=True,ASR_updates=0,seconds=time.monotonic()-start))
