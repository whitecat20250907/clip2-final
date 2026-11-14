import os,sys,pathlib,time,csv,itertools
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from head_tools_v1 import *
def wordnorm(ii):
    a = support[ii][legal[ii]]
    med = np.nan_to_num(np.nanmedian(a, axis=0))
    v = np.where(np.isfinite(a), a, med)
    mu = v.mean(0)
    sd = np.maximum(v.std(0), 0.0001)
    return dict(median=med, mean=mu, std=sd)

def make_word_extra(norm, misalign=False):
    z = np.clip((np.where(np.isfinite(support), support, norm['median']) - norm['mean']) / norm['std'], -4, 4)
    if misalign:
        z = z.copy()
        for (j, n) in enumerate(legal.sum(1)):
            n = int(n)
            z[j, :n, 2:4] = z[j, :n, 2:4][::-1]
            z[j, :n, 4:6] = np.roll(z[j, :n, 4:6], 1, axis=0)
    mu = (z * legal[..., None]).sum(1) / legal.sum(1)[:, None]
    std = np.sqrt(np.maximum(((z - mu[:, None, :]) ** 2 * legal[..., None]).sum(1) / legal.sum(1)[:, None], 0))
    marg = np.c_[mu, std, (missing * legal).sum(1) / legal.sum(1)]
    cov = []
    diff = []
    weak = []
    for (a, b) in pairs:
        cov.append(((z[..., a] - mu[:, None, a]) * (z[..., b] - mu[:, None, b]) * legal).sum(1) / legal.sum(1))
        diff.append((abs(z[..., a] - z[..., b]) * legal).sum(1) / legal.sum(1))
        weak.append(((z[..., a] < -0.5) & (z[..., b] < -0.5) & legal).sum(1) / legal.sum(1))
    cov = np.asarray(cov).T
    diff = np.asarray(diff).T
    weak = np.asarray(weak).T
    return {'marginal13': marg, 'sameword_cov25': np.c_[marg, cov], 'sameword_full49': np.c_[marg, cov, diff, weak]}
mode=sys.argv[1];assert mode in ['band','temporal','sameword'];start=time.monotonic();ctx=load_context(R);globals().update(ctx);N=len(fitids);ix={s:i for i,s in enumerate(ss)}
def waitfeature():
 if mode=='sameword':return
 while True:
  if mode=='temporal':
   f=R/'TEMPORAL_FEATURE_DRIVER_TERMINAL_V1.json'
   if f.exists():assert read(f)['actual_exit_code']==0,('temporalfeaturefailed',str(f))
   good=(R/'TEMPORAL_FEATURES_COMPLETE_V1.json').exists()
  else:
   good=True
   for part in [0,1]:
    f=R/('part%d'%part)/'BAND_DRIVER_TERMINAL_V1.json'
    if f.exists():assert read(f)['actual_exit_code']==0,('bandfeaturefailed',str(f))
    else:good=False
  if good:return
  if time.monotonic()-start>10800:raise RuntimeError('Featurewaitexceeded')
  time.sleep(10)
def loadfull():
 if mode=='sameword':
  norm=wordnorm(np.arange(NN));save(R/'FULL_WORD_SUPPORT_NORMALIZER_V1.npz',**norm);out=make_word_extra(norm);save(R/'SAMEWORD_ALL20239_PUBLIC_FEATURES_V1.npz',signals=np.asarray(ss),**out);return out
 if mode=='temporal':
  f=R/'TEMPORAL_ALL20239_PUBLIC_FEATURES_V1.npz';assert sha(f)==read(R/'TEMPORAL_FEATURES_COMPLETE_V1.json')['feature_sha']
  with np.load(f,allow_pickle=False)as z:assert z['signals'].tolist()==ss;return {k:z[k].copy()for k in ['temporal192','shuffled128']}
 rows={s:microband[j]for j,s in enumerate(micro)};receipts=[]
 for part in [0,1]:
  root=R/('part%d'%part);complete=read(root/'BAND_PART_COMPLETE_V1.json')
  for rec in complete['receipts']:
   f=root/rec['file'];assert sha(f)==rec['sha256'];receipts.append(dict(path=str(f),sha256=rec['sha256']))
   with np.load(f,allow_pickle=False)as z:
    assert z['signals'].tolist()==rec['signals']
    for j,s in enumerate(rec['signals']):assert s not in rows;rows[s]=z['band24'][j].copy()
 assert set(rows)==set(ss);band=np.asarray([rows[s]for s in ss]);assert band.shape==(20239,24)and not np.isinf(band).any()
 save(R/'BAND24_ALL20239_PUBLIC_FEATURES_V1.npz',signals=np.asarray(ss),band24=band)
 put(R/'BAND_FEATURES_COMPLETE_V1.json',dict(rows=len(ss),feature_sha=sha(R/'BAND24_ALL20239_PUBLIC_FEATURES_V1.npz'),ASR_updates=0,source_ASR44_SHARED=True,receipts=receipts))
 return {'band24':band}
if mode=='band':
 with np.load(p['micro_features'],allow_pickle=False)as z:assert z['signals'].tolist()==micro;microband=z['stability72'][:,48:72].copy()
 families=[('band24',445),('band24',573)];mextra={'band24':microband};all_extra=None;sourcefeature=p['micro_features']
elif mode=='sameword':
 sourcefeature=str(pathlib.Path(C['PREV'])/'PUBLIC_RAW_SUPPORT_AND_BASES_V1.npz')
 with np.load(sourcefeature,allow_pickle=False)as z:
  assert z['signals'].tolist()==ss;support=z['support'].copy();legal=z['legal'].copy();missing=z['missing'].copy()
 groups=[0,0,1,1,2,2];pairs=[(a,b)for a,b in itertools.combinations(range(6),2)if groups[a]!=groups[b]];assert len(pairs)==12
 norm=wordnorm(ti[:N]);assert (R/'MICRO_WORD_SUPPORT_NORMALIZER_V1.npz').exists();all_micro=make_word_extra(norm);families=[(k,b)for k in ['sameword_cov25','sameword_full49']for b in [445,573]];mextra={k:v[ti]for k,v in all_micro.items()};all_extra=None
else:
 waitfeature();all_extra=loadfull();families=[('temporal192',573),('shuffled128',573)];mextra={k:v[ti]for k,v in all_extra.items()};sourcefeature=str(R/'TEMPORAL_ALL20239_PUBLIC_FEATURES_V1.npz')
sealed=read(R/'OBSERVABLE_MICRO_SEALED_V1.json');assert sha(R/'OBSERVABLE_MICRO_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz')==sealed['prediction_sha']
with np.load(R/'OBSERVABLE_MICRO_HELD_PREDICTIONS_BEFORE_METRICS_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==held;pred={k:z[k].copy()for k in z.files if k!='signals'}
models=sealed['models'];candidates=[]
for m in models:
 assert sha(R/m['file'])==m['sha'];stem=m['file'][6:-7]
 for shrink in [1.,.5]:
  key=stem+'_sh%d'%round(100*shrink);assert key in pred;candidates.append(dict(key=key,family=m['family'],base_dim=m['base_dim'],kind=m['kind'],new_lambda=m['new_lambda'],shrink=shrink,control='control%d_%s'%(m['base_dim'],m['kind'])))
print('REUSE_MICRO_MODELS_AND_PREDICTIONS',len(models),flush=True)
mm={k:metric(v,yy[N:])for k,v in pred.items()}
for c in candidates:c['gain']=mm[c['control']]['RMSE']-mm[c['key']]['RMSE']
selected=[]
for kind in ['sigmoid','linear']:
 possible=[c for c in candidates if c['kind']==kind]
 if not possible:continue
 best=max(possible,key=lambda c:c['gain'])
 if best['gain']>.005:selected.append(best)
if mode=='sameword'and selected:
 c=selected[0];bad=make_word_extra(norm,True)[c['family']][ti];base=bases[44][ti]if c['base_dim']==445 else np.c_[bases[44][ti],h14[ti]];xx=np.c_[base,bad];st=fit(xx[:N],yy[:N],c['base_dim'],'sigmoid',c['new_lambda']);save(R/'MICRO_MISALIGNED_CONTROL_V1.npz',**st);qq=predict(st,xx[N:]);qq=c['shrink']*qq+(1-c['shrink'])*controls[(c['base_dim'],'sigmoid')];save(R/'MICRO_MISALIGNED_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.asarray(held),prediction=qq);mm['matched_misaligned_control']=metric(qq,yy[N:])
microres=dict(mode=mode,metrics=mm,candidates=candidates,selected=selected,TRAIN_FIT_rows=N,TRAIN_held_rows=len(held),conditional_not_wholepipeline_OOF=True,temporal_shuffle_control_not_proof_of_order_benefit=True,seconds=time.monotonic()-start)
put(R/'OBSERVABLE_MICRO_RESULT_V1.json',microres);print('OBSERVABLE_MICRO_SELECTED',mode,json.dumps(selected),flush=True)
if not selected:
 put(R/'OBSERVABLE_READOUT_COMPLETE_V1.json',dict(code=0,promoted=False,seconds=time.monotonic()-start));raise SystemExit(0)
waitfeature()
if all_extra is None:all_extra=loadfull()
wp=read(pathlib.Path(C['WIN'])/'JOINT_POOL_FULL_PLAN_V1.json')
with np.load(wp['parent_prediction'],allow_pickle=False)as z:assert z['signals'].tolist()==dev;parents={s:z['paired5_seed%d'%s].copy()for s in [44,45]}
L=pathlib.Path(C['BASELINE_LINK_ROOT'])
assert sha(L/'LINK_FULL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz')==read(L/'FINAL_INDEPENDENT_REPLAY_V1.json')['prediction_sha']
with np.load(L/'LINK_FULL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz',allow_pickle=False)as z:
 assert z['signals'].tolist()==dev;baseline={s:z['outside30_sh50_seed%d'%s].copy()for s in [44,45]};q445={s:z['rawCount_seed%d'%s].copy()for s in [44,45]}
H14=pathlib.Path(C['H14_ROOT'])
with np.load(H14/'H14_AND_FIXED_DEPTH_AVERAGE_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz',allow_pickle=False)as z:assert z['signals'].tolist()==dev;h14parent={s:z['outside30_sh100_seed%d'%s].copy()for s in [44,45]}
goal=pathlib.Path(C['NEW_GOAL_ROOT']);sf=goal/'SECONDARY_ALL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz';assert sha(sf)==read(goal/'SECONDARY_MODELS_AND_PREDICTIONS_SEALED_V1.json')['predictions_sha']
with np.load(sf,allow_pickle=False)as z:assert z['signals'].tolist()==dev;secondary={s:z['singleGLM573_seed%d'%s].copy()for s in [44,45]}
full={};fullmodels=[];priority=[]
for number,c in enumerate(selected):
 family,base_dim,kind,lam=c['family'],c['base_dim'],c['kind'],c['new_lambda'];stem='candidate%d_%s_base%d_%s'%(number,family,base_dim,kind);extra=all_extra[family]
 for seed in [44,45]:
  base=bases[seed]if base_dim==445 else np.c_[bases[seed],h14];x=np.c_[base,extra];st=fit(x[:NN],Y,base_dim,kind,lam);fn=stem+'_FULL_SEED%d_V1.npz'%seed;save(R/fn,**st);q=predict(st,x[NN:])
  qh=(h14parent[seed]-.7*parents[seed])/.3;assert abs(.7*parents[seed]+.15*q445[seed]+.15*qh-baseline[seed]).max()<1e-10
  direct=.7*parents[seed]+.3*q;replace=.7*parents[seed]+.15*q+( .15*q445[seed]if base_dim==573 else .15*qh)
  full[stem+'_rawCount_seed%d'%seed]=q;full[stem+'_direct30_seed%d'%seed]=direct;full[stem+'_replace15_seed%d'%seed]=replace;full[stem+'_replace15half_seed%d'%seed]=.5*replace+.5*baseline[seed];full[stem+'_halfWithSecondary_seed%d'%seed]=.5*direct+.5*secondary[seed]
  fullmodels.append(dict(seed=seed,file=fn,sha=sha(R/fn),family=family,base_dim=base_dim,kind=kind,new_lambda=lam,TRAIN_rows=NN,dimension=x.shape[1],stem=stem));print('OBSERVABLE_FULL_FIT',mode,stem,seed,round(time.monotonic()-start,1),flush=True)
 for method in ['direct30','replace15','replace15half','halfWithSecondary']:
  key=stem+'_'+method;full[key+'_mean']=(full[key+'_seed44']+full[key+'_seed45'])/2;priority.append(key+'_mean')
full['baseline_mean']=(baseline[44]+baseline[45])/2;full['secondary_mean']=(secondary[44]+secondary[45])/2
save(R/'OBSERVABLE_FULL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz',signals=np.asarray(dev),**full)
featurefile=R/'SAMEWORD_ALL20239_PUBLIC_FEATURES_V1.npz'if mode=='sameword'else R/('TEMPORAL_ALL20239_PUBLIC_FEATURES_V1.npz'if mode=='temporal'else'BAND24_ALL20239_PUBLIC_FEATURES_V1.npz')
put(R/'OBSERVABLE_FULL_SEALED_V1.json',dict(models=fullmodels,priority=priority,TRAIN_ID_SHA=digest(train),DEV_ID_SHA=digest(dev),feature_sha=sha(featurefile),feature_file=str(featurefile),predictions_sha=sha(R/'OBSERVABLE_FULL_DEV_PREDICTIONS_BEFORE_GOLD_V1.npz'),formal_DEV_gold_read_THIS_RUN=False,ASR_updates=0))
assert sha(bp['truth'])==bp['truth_sha']
with pathlib.Path(bp['truth']).open(newline='')as f:rs=list(csv.DictReader(f))
assert[r['signal']for r in rs]==dev;truth=np.asarray([float(r['truth'])for r in rs]);fm={k:metric(v,truth)for k,v in full.items()};assert abs(fm['baseline_mean']['RMSE']-25.886260793636932)<1e-8 and abs(fm['secondary_mean']['RMSE']-25.878525391096886)<1e-8;best=min(priority,key=lambda k:fm[k]['RMSE']);passing=[k for k in priority if fm[k]['RMSE']<25.85]
res=dict(mode=mode,metrics=fm,priority=priority,best=best,passing=passing,goal_strict_under25_85=bool(passing),recommended_RMSE=fm[best]['RMSE'],recommended_PCC=fm[best]['PCC'],DEV_repeated_development=True,EvalAI_verified=False,TRAIN_rows=NN,DEV_ID_SHA=digest(dev),ASR_updates=0,seconds=time.monotonic()-start)
put(R/'OBSERVABLE_FULL_ORIGINAL4059_RESULT_V1.json',res);print('OBSERVABLE_FULL_RESULT',json.dumps(res),flush=True)
put(R/'OBSERVABLE_READOUT_COMPLETE_V1.json',dict(code=0,promoted=True,seconds=time.monotonic()-start))
