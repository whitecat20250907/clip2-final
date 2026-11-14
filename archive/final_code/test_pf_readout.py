"""Original measured PRIOR_FIXED graph; frozen heads only, no optimization."""
import ast,csv,hashlib,json,pathlib
import numpy as np
from scipy.special import expit,logit
import torch
B=pathlib.Path(__file__).parent;V=B.parents[2];J=pathlib.Path('E:/icassp/zhaojiahao/jobs');C=B/'outputs/TEST_components';SEEDS=[142,242,342,442]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def csvp(p):
 rows=list(csv.reader(p.open(newline='')));return np.asarray([r[0]for r in rows]),np.asarray([float(r[1])/100 for r in rows])
def defs(path,names,scope):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
def historical_csv_roundtrip(q):return np.asarray([float(format(x*100,'.10f'))/100 for x in q])
def music_root(p2,mf55,mf72,moss72):
 # Preserve the historical float32 intermediate and CSV boundaries.
 mfown=historical_csv_roundtrip(.9*p2.astype(np.float32)+.1*mf55).astype(np.float32).astype(float)
 mossbase=historical_csv_roundtrip(.9*mfown+.1*moss72)
 return historical_csv_roundtrip(mossbase+.09*(mf72-mf55))
def old_mf(raw,mask,proof):
 vals=[];valid=arr(V/'new_angles_20261021/music_witness_full_train/readout/valid.npz')
 for seed in SEEDS:
  p=V/f'new_angles_20261021/music_witness_full_train/readout/full/witness_full_s{seed}/model.pt';cp=torch.load(p,map_location='cpu',weights_only=False);assert cp['dim']==55
  net=torch.nn.Sequential(torch.nn.Linear(55,8),torch.nn.Tanh(),torch.nn.Linear(8,1));net.load_state_dict({k.removeprefix('head.'):v for k,v in cp['state'].items()});net.eval().requires_grad_(False)
  def infer(x,m):
   x=((x-cp['mean'])/cp['std']).astype(np.float32);x[~m.astype(bool)]=0;out=[]
   with torch.inference_mode():
    for i in range(0,len(x),256):
     mm=torch.tensor(m[i:i+256],dtype=torch.float32);out.append(((net(torch.from_numpy(x[i:i+256])).squeeze(-1).sigmoid()*mm).sum(1)/mm.sum(1)).numpy())
   return np.concatenate(out)
  # Cross-backend float32 GEMM/tanh reduction: bounded below 5e-7 probability.
  # This ancestor cancels algebraically; it remains solely to reproduce the
  # original float32/CSV intermediate operations, not as a new model member.
  gap=float(abs(infer(valid['x'],valid['mask'])-arr(p.with_name('valid.npz'))['prediction']).max());assert gap<5e-7,(seed,gap)
  proof.append(dict(role='original_MF55_numeric_ancestor',seed=seed,SHA=sha(p),VALID_maxgap_probability=gap));vals.append(infer(raw,mask))
 return np.mean(vals,0)
def er_prediction(x,mask,sev,parent,root,proof):
 out=[];valid=arr(root/'valid.npz')
 for seed in SEEDS:
  p=root/f'results/true_s{seed}/model.pt';cp=torch.load(p,map_location='cpu',weights_only=False);selected=cp['selected_interactions']
  def infer(xx,mm,ss,pp):
   vv=(xx-cp['mean'])/cp['std'];vv[~mm.astype(bool)]=0
   design=np.concatenate([np.ones((*xx.shape[:2],1),np.float32),vv,np.broadcast_to((ss==1)[:,None,None],(*xx.shape[:2],1)),np.broadcast_to((ss==2)[:,None,None],(*xx.shape[:2],1)),vv[:,:,selected]*(ss==1)[:,None,None],vv[:,:,selected]*(ss==2)[:,None,None]],2).astype(np.float32)
   with torch.inference_mode():
    q=torch.from_numpy(design);m=torch.tensor(mm,dtype=torch.float32);shift=.2*torch.tanh(torch.nn.functional.linear(q,cp['weight']).squeeze(-1));shift=(shift*m).sum(1)/m.sum(1);base=torch.tensor(pp,dtype=torch.float32);return torch.sigmoid(torch.logit(base.clamp(1e-6,1-1e-6))+shift).numpy()
  q=infer(valid['x'],valid['mask'],valid['severity'],valid['parent']);gap=float(abs(q-arr(p.with_name('valid.npz'))['prediction']).max());assert gap<2e-7
  proof.append(dict(role=root.name,seed=seed,SHA=sha(p),VALID_maxgap_probability=gap));out.append(infer(x,mask,sev,parent))
 return np.mean(out,0)
def main():
 torch.set_num_threads(2);assets=read(B/'original_assets/ASSET_INDEX.json');pins={r['original_path']:r['SHA']for r in assets['copied_files']}
 for p,h in pins.items():
  if p.endswith('model.pt') and ('evidence_relations_20261022' in p or 'music_witness_full_train/readout/full' in p.replace('\\','/')):assert sha(pathlib.Path(p))==h
 src=J/'clip2_fb_parent_replay_20261024.py';assert sha(src)=='d2b940206929d4749b670cfdc0af78997060fca2152c60a39ea489a719edf4bc';env=dict(np=np,expit=expit,logit=logit);defs(src,['anchor','meanword','standard','sentence_apply','word_apply','z2_apply','pipeline'],env)
 defs(J/'v10_ER01_relations_pipeline_20261022.py',['relations'],env)
 cfg=read(V/'next_fixed_blocks_20261024/train_parent_audit/frozen_node_configuration.json');models=cfg['node_models'];coef=cfg['first_composition_coefficients'];fixed=cfg['final_PRIOR_FIXED_model'];z=arr(V/'fourth_round_20261023/V11_replay/joint_fit_inputs_conditional.npz')
 validMF=arr(V/'next_optimization_20261021/G01_MF/readout/valid.npz');validMO=arr(V/'next_optimization_20261021/G01_G02/G01_valid.npz');vrel=env['relations'](validMF['x'],validMO['x']);vrel[~validMF['mask'].astype(bool)]=0;erv=arr(V/'evidence_relations_20261022/ER01/valid.npz');assert np.array_equal(vrel,erv['x'])
 def cached_mean(pattern):return np.mean([arr(V/(pattern.format(seed=seed)))['prediction']for seed in SEEDS],0)
 old=cached_mean('new_angles_20261021/music_witness_full_train/readout/full/witness_full_s{seed}/valid.npz');mf=cached_mean('next_optimization_20261021/G01_MF/readout/results/witness_full_s{seed}/valid.npz');mo=cached_mean('next_optimization_20261021/G01_G02/results/G01_s{seed}/valid.npz')
 # VALID parent is the exact original float32 P2 cache, avoiding new rounding.
 root=music_root(validMF['parent'],old,mf,mo);assert np.array_equal(root,z['valid_parent_frozen587135'])
 z['valid_parent_frozen587135']=root;validpf,stages=env['pipeline'](z,models,coef,'valid',erv['parent'].astype(float),fixed);_,gold=csvp(V/'third_round_20261023/Z13_attribution_20261023/PRIOR_FIXED/candidate/valid_predictions.csv');pf_gap=float(abs(validpf-gold).max()*100);assert pf_gap<1e-6
 reconstruction=dict(root_VALID2117_bitwise_exact=True,PRIOR_FIXED_VALID_maxgap_points=pf_gap,MOSS_actual_arm='G01',MOSS_actual_dim=72,MOSS_parameters_per_head=593,MOSS_handoff_G02_rejected_as_wrong_ancestry=True,original_fusion_ratios_unchanged=True,training_updates=0)
 (B/'outputs/TEST_MOSS_ANCESTRY_GRAPH_QUALIFICATION.json').write_text(json.dumps(reconstruction,indent=2));print(json.dumps(reconstruction),flush=True)
 mf=arr(C/'MF_word_features.npz');mo=arr(C/'MOSS_word_features.npz');native=arr(C/'ER02_temporal36.npz');static=arr(C/'PF_static_features.npz');ids=mf['signal_ids'];mask=mf['mask'];assert all(np.array_equal(q['signal_ids'],ids)for q in [mo,native,static]);assert np.array_equal(mask,mo['mask'])and np.array_equal(mask,native['mask']);rows=read(B/'outputs/TEST_LMY_NATIVE/selection.json')[2:];assert [r['signal']for r in rows]==ids.tolist();sev=np.asarray([r['severity']for r in rows]);proof=[]
 p2ids,p2=csvp(C/'P2_EVAL_intermediate.csv');assert np.array_equal(ids,p2ids);old=old_mf(mf['x55'],mask,proof);root=music_root(p2,old,mf['mean_prediction'],mo['mean_prediction']);relations=env['relations'](mf['x72'],mo['x72']);relations[~mask]=0
 er1=er_prediction(relations,mask,sev,root,V/'evidence_relations_20261022/ER01',proof);er2=er_prediction(np.concatenate([relations,native['x']],-1),mask,sev,root,V/'evidence_relations_20261022/expanded/ER02',proof)
 q=arr(C/'word_proxy.npz');assert np.array_equal(q['signal_ids'],ids) and np.array_equal(q['mask'],mask);proxy=q['proxy'];vgap=read(C/'word_proxy.json')['VALID2117_maxgap'];assert vgap==0.
 ez={('valid_'+k):v for k,v in static.items()if k!='signal_ids'};ez.update(valid_parent_frozen587135=root,valid_proxy=proxy,valid_mask=mask,valid_ER01_frozen_cache=er1,valid_ER02_frozen_cache=er2)
 prediction,stages=env['pipeline'](ez,models,coef,'valid',root.astype(np.float32).astype(float),fixed);assert np.isfinite(prediction).all()and((prediction>=0)&(prediction<=1)).all()
 p=C/'PRIOR_FIXED_EVAL_intermediate.npz'
 with p.open('xb')as f:np.savez_compressed(f,signal_ids=ids,prediction=prediction,mask=mask,proxy=proxy,**stages)
 report=dict(rows=2189,output_SHA=sha(p),VALID_graph=reconstruction,VALID_proxy_maxgap=vgap,readout_qualification=proof,training_updates=0,final_joint_complete=False,AB_input_source_review_pending=True)
 (C/'PRIOR_FIXED_EVAL_intermediate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
