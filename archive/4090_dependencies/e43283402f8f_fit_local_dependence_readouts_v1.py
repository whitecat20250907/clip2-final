
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common_v1 import *
import torch,math,gc
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
assert read(R/'FEATURE_TERMINAL_V1.json')['exitcode']==0
fs=read(R/'LOCAL_DEPENDENCE_ALL_PUBLIC_FEATURES_SEALED_BEFORE_TARGET_FITS_V1.json');fp=R/'LOCAL_DEPENDENCE_PUBLIC5120_BEFORE_NEW_TARGET_FITS_V1.npz';assert sha(fp)==fs['feature_sha']
pub=nz(fp);assert pub['signals'].tolist()==mids and pub['features'].shape==(5120,12,26)
legal=pub['legal'];raw=pub['features'];common=np.concatenate([raw[...,:6],raw[...,18:26]],-1);assert common.shape==(5120,12,14)
extras=dict(common_original_physical=np.concatenate([common,np.zeros((5120,12,6))],-1),wrong_location=np.concatenate([common,raw[...,12:18]],-1),target_location=np.concatenate([common,raw[...,6:12]],-1))
MAIN=R.parent/'full_native_path_word_mean_Brier_micro_v1';LAT=R.parent/'latent_response_joint_word_count_micro_v1';PA=R.parent/'readout_parent_strength_inference_audit_v1';ANCHOR=R.parent/'native_word_grounded_encoder_joint_target_micro_v1'
ls=read(LAT/'LATENT_INPUT_PREPARATION_COMPLETE_V1.json');af=LAT/'LATENT_MICRO_PUBLIC_INPUTS_AND_FIT_ONLY_NORMALIZERS_V1.npz';assert sha(af)==ls['public_inputs_sha'];a=nz(af)
assert a['signals'].tolist()==mids and np.array_equal(legal,a['legal'])and not set(a['groups'][:nf])&set(a['groups'][nf:])
anchorfile=ANCHOR/'PUBLIC_FROZEN_WARM_ANCHOR5120_BEFORE_NEW_UPDATES_V1.npz';aq=read(ANCHOR/'ACTUAL_NATIVE_WORD_EVENT_ENCODER_TASK_QUALIFICATION_BEFORE_OPTIMIZER_V1.json');assert sha(anchorfile)==aq['anchor_sha'];anch=nz(anchorfile)
assert anch['signals'].tolist()==mids and np.array_equal(anch['legal'],legal);anchor_eta=anch['eta'];anchor_q=anch['q']
pr=nz(PA/'FROZEN_READOUT_FULL_PARENT_HELD_PREDICTIONS_BEFORE_TARGETS_V1.npz');assert pr['signals'].tolist()==mids[nf:];assert abs(anchor_q[nf:]-pr['warm_full_parent']).max()<2e-7
warm=nz(MAIN/'MICRO5120_NATIVE_MEAN_PUBLIC_FEATURES_BEFORE_NEW_COUNT_TARGETS_V1.npz');wn=nz(MAIN/'NATIVE_MEAN_WORD57_FIT_ONLY_NORMALIZERS_BEFORE_UPDATES_V1.npz');assert warm['signals'].tolist()==mids
wextra=np.where(legal[...,None],np.clip((warm['warm_word']-wn['warm_mean'])/wn['warm_std'],-4,4),0).astype(np.float32)
x57=np.concatenate([a['x'],wextra],-1);assert x57.shape==(5120,12,57)
cpfile=MAIN/'NATIVE_MEAN_WORD57_warm_SEED44_FINAL6_V1.pt';cp=torch.load(cpfile,map_location='cpu',weights_only=False)
W1=cp['weights']['branch.0.weight'].float().detach();b1=cp['weights']['branch.0.bias'].float().detach();W2=cp['weights']['branch.2.weight'].float().detach();b2=cp['weights']['branch.2.bias'].float().detach()
with torch.no_grad():
 oldh=F.linear(torch.as_tensor(x57),W1,b1);base_d=2*torch.tanh(F.linear(F.gelu(oldh),W2,b2).squeeze(-1)/2)
assert not any(v.requires_grad for v in [W1,b1,W2,b2,oldh,base_d])
normalized={};norms={}
for arm,e in extras.items():
 vals=e[:nf][legal[:nf]];mu=vals.mean(0);sd=np.maximum(vals.std(0),1e-4)
 normalized[arm]=np.where(legal[...,None],np.clip((e-mu)/sd,-4,4),0).astype(np.float32)
 norms[arm+'_mean']=mu;norms[arm+'_std']=sd
save('LOCAL_DEPENDENCE_WORD_FIT_ONLY_NORMALIZERS_BEFORE_NEW_UPDATES_V1.npz',**norms)
tf=LAT/'LATENT_TRAIN_ONLY_WORD_AND_COUNT_TARGETS_V1.npz';assert sha(tf)==ls['training_targets_sha']
with np.load(tf,allow_pickle=False)as z:
 assert z['signals'].tolist()==mids
 targets={k:z[k][:nf].copy()for k in ['wordy','known','K','NW']}
assert np.array_equal(targets['NW'],legal[:nf].sum(1))
WEIGHTS=torch.ones(1,dtype=torch.float64)
nodes=[n for n in ast.parse((LAT/'latent_joint_core_v2_quota_recovered.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='partial_joint_logprob'];assert len(nodes)==1
exec(compile(ast.Module(body=nodes,type_ignores=[]),'retained_partial_joint','exec'),globals())
class WordAdapter(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.A=nn.Parameter(torch.zeros((16,20)))
 def forward(self,ii):
  ii=np.asarray(ii,int);e=torch.as_tensor(normalized[self.arm][ii]);h=oldh[ii]+F.linear(e,self.A)
  new=2*torch.tanh(F.linear(F.gelu(h),W2,b2).squeeze(-1)/2);d=new-base_d[ii]
  ll=torch.as_tensor(legal[ii]);eta=torch.where(ll,(torch.as_tensor(anchor_eta[ii])+d).clamp(-20,20),torch.zeros_like(d))
  p=eta.double().sigmoid();q=(p*ll).sum(1)/ll.sum(1)
  return q,p,d,ll
 def objective(self,ii):
  ii=np.asarray(ii,int);assert ii.max()<nf
  q,p,d,ll=self(ii);wy=torch.as_tensor(targets['wordy'][ii]);known=torch.as_tensor(targets['known'][ii]);K=torch.as_tensor(targets['K'][ii])
  primary=(-partial_joint_logprob(p[...,None],ll,wy,known,K,WEIGHTS)/ll.sum(1)).mean()
  return primary+.001*(d.double().square()*ll).sum()/ll.sum()
qualification=[]
for arm in extras:
 m=WordAdapter(arm);q,p,d,ll=m(np.arange(32));gap=float(abs(q.detach().numpy()-anchor_q[:32]).max());assert gap<2e-7
 qh=m(np.arange(nf,nf+32))[0];heldgap=float(abs(qh.detach().numpy()-anchor_q[nf:nf+32]).max());assert heldgap<2e-7
 loss=m.objective(np.arange(32));loss.backward();gn=float(m.A.grad.norm());assert math.isfinite(gn)and gn>0
 qualification.append(dict(arm=arm,initial_FIT_gap=gap,initial_public_held_gap=heldgap,actual_task_gradient_norm=gn,trainable_parameters=m.A.numel()))
 del m
put('LOCAL_WORD_ADAPTER_ACTUAL_TASK_QUALIFICATION_BEFORE_OPTIMIZER_V1.json',dict(qualified=True,checks=qualification,feature_sha=fs['feature_sha'],anchor_sha=sha(anchorfile),frozen_head_sha=sha(cpfile),new_optimizer_updates=0,no_held_targets_for_qualification=True))
pred=dict(existing_warm_word=anchor_q[nf:]);models=[];wordpred={};start=time.monotonic()
for arm in extras:
 m=WordAdapter(arm);opt=torch.optim.AdamW([m.A],lr=.001,weight_decay=.01);steps=0;trace=[]
 for epoch in range(6):
  order=np.random.default_rng(20261022+epoch).permutation(nf);losses=[]
  for off in range(0,nf,128):
   ii=order[off:off+128];opt.zero_grad(set_to_none=True);loss=m.objective(ii);assert torch.isfinite(loss);loss.backward();assert torch.isfinite(m.A.grad).all();torch.nn.utils.clip_grad_norm_([m.A],1.);opt.step();steps+=1;losses.append(float(loss.detach()))
  trace.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),A_norm=float(m.A.detach().norm())));note(stage='local_word_adapter_epoch',arm=arm,**trace[-1],seconds=time.monotonic()-start)
 assert steps==192
 fn='WORD_LOCAL_'+arm+'_SEED44_FINAL6_V1.pt'
 with(R/fn).open('xb')as f:torch.save(dict(weights=m.state_dict(),optimizer=opt.state_dict(),steps=steps,epochs=6,seed=44,FIT_signals=np.array(mids[:nf]),history=trace,normalizer_sha=sha(R/'LOCAL_DEPENDENCE_WORD_FIT_ONLY_NORMALIZERS_BEFORE_NEW_UPDATES_V1.npz'),anchor_sha=sha(anchorfile),frozen_head_sha=sha(cpfile)),f)
 qs=[];ps=[]
 with torch.no_grad():
  for off in range(nf,len(mids),128):
   ii=np.arange(off,min(off+128,len(mids)));q,p,_,ll=m(ii);qs.append(q.numpy());ps.append(np.where(ll.numpy(),p.numpy(),0))
 name='Word_'+arm;pred[name]=np.concatenate(qs);wordpred[name+'_words']=np.concatenate(ps);pred[name+'_half']=.5*pred[name]+.5*pred['existing_warm_word']
 models.append(dict(arm=name,file=fn,sha=sha(R/fn),parameters=320,steps=steps));del m,opt;gc.collect()
# Fixed Count measurement comparison; no held metrics have been inspected.
cc=nz(G/'lowrank_cross_evidence_Count_v1/COUNT_PUBLIC_MATRICES_V1.npz',keys=['signals','fullX44']);assert cc['signals'].tolist()==ss;ii=np.array([idx[s]for s in mids]);X=cc['fullX44'][ii]
basefile=MAIN/'CAL_Count920_baseline_SEED44_V1.npz';st=nz(basefile);pen=st['penalty'][1:];assert len(pen)==920
pred['existing_Count920']=count_pred(X[nf:],st);baseline=nz(MAIN/'CAL_Count_TRAIN_HELD1024_PREDICTIONS_BEFORE_SCORING_V1.npz');assert baseline['signals'].tolist()==mids[nf:];assert abs(pred['existing_Count920']-baseline['baseline']).max()<1e-12
p=G/'paired_word_moment_count_refit_v2/fit_pooled_pair_v1.py';node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='fit_count');exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),globals())
with np.load(bp['count_targets'],allow_pickle=False)as z:
 assert z['signals'].tolist()==train;Yfit=z['y'][ii[:nf]]
def pool(v):
 return np.array([np.concatenate([fn(v[j][legal[j]],axis=0)for fn in [np.mean,np.std,np.min,np.max]])for j in range(len(v))])
pools=dict(common56=pool(common),wrong80=pool(extras['wrong_location']),target80=pool(extras['target_location']))
for arm,e in pools.items():
 xx=np.c_[X,e];assert np.isfinite(xx[:nf]).any(0).all();fit=fit_count(xx[:nf],Yfit,np.r_[pen,np.full(e.shape[1],1000.*nf/len(train))])
 fn='COUNT_LOCAL_'+arm+'_SEED44_FIT4096_V1.npz';save(fn,**fit,FIT_signals=np.array(mids[:nf]),source_feature_sha=np.array(fs['feature_sha']))
 pred['Count_'+arm]=count_pred(xx[nf:],fit);models.append(dict(arm='Count_'+arm,file=fn,sha=sha(R/fn),new_channels=e.shape[1]));note(stage='local_count_measurement_fit_complete',arm=arm,seconds=time.monotonic()-start)
for arm in ['wrong80','target80']:
 k='Count_'+arm;pred[k+'_half']=.5*(pred[k]+pred['existing_Count920'])
 pred[k+'_paired_half_on_warm']=expit(logit(pred['existing_warm_word'].clip(1e-6,1-1e-6))+.5*(logit(pred[k].clip(1e-6,1-1e-6))-logit(pred['Count_common56'].clip(1e-6,1-1e-6))))
assert all(len(v)==1024 and np.isfinite(v).all()for v in pred.values())
save('LOCAL_DEPENDENCE_ALL_HELD1024_PREDICTIONS_BEFORE_METRICS_V1.npz',signals=np.array(mids[nf:]),**pred,**wordpred)
put('LOCAL_DEPENDENCE_ALL_MODELS_PREDICTIONS_SEALED_BEFORE_METRICS_V1.json',dict(models=models,predictions_sha=sha(R/'LOCAL_DEPENDENCE_ALL_HELD1024_PREDICTIONS_BEFORE_METRICS_V1.npz'),feature_sha=fs['feature_sha'],anchor_sha=sha(anchorfile),source_sha=sha(R/'fit_local_dependence_readouts_v1.py'),all_new_fits_FIT4096_only=True,no_DEV_gold=True,wholepipelineOOF=False))
with np.load(bp['count_targets'],allow_pickle=False)as z:Yheld=z['y'][ii[nf:]]
metrics={k:dict(RMSE=float(100*np.sqrt(np.mean((v-Yheld)**2))),PCC=float(np.corrcoef(v,Yheld)[0,1]))for k,v in pred.items()}
for k,v in metrics.items():
 v['gain_vs_warm_word']=metrics['existing_warm_word']['RMSE']-v['RMSE'];v['gain_vs_Count920']=metrics['existing_Count920']['RMSE']-v['RMSE']
put('LOCAL_DEPENDENCE_TRAIN_HELD_RESULT_V1.json',dict(role='TRAIN4096FIT1024Audio2groupheld_NOTformal4059DEV',metrics=metrics,primary='Word_target_location',matched_control='Word_wrong_location',common_control='Word_common_original_physical',secondary_Count_and_fixed_complements=True,no_ASR_updates=True,word_adapter_trainable320=True,no_DEV_gold=True,wholepipelineOOF=False,official_best_unchanged=15.713360199282489,seconds=time.monotonic()-start))
note(stage='local_dependence_result',metrics=metrics,seconds=time.monotonic()-start)
