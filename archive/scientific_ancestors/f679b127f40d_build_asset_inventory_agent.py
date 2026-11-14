"""Inventory only: no model forward/fit, copying, deployment or network."""
from pathlib import Path
import json,hashlib,datetime,numpy as np
V=Path(r'E:\icassp\zhaojiahao\CLIP2-new\algorithms\03_my_improvements\experiments\v10_20261016')
S=V/'final_selective_risk_20261027';D=S/'delivery_preparation';N=V/'next_jury_20261024'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),SHA=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def refs(v):
 o=[]
 if isinstance(v,dict):
  if isinstance(v.get('path'),str)and isinstance(v.get('SHA'),str)and len(v['SHA'])==64:o.append(dict(path=v['path'],SHA=v['SHA']))
  for x in v.values():o+=refs(x)
 elif isinstance(v,list):
  for x in v:o+=refs(x)
 return o
inventory=read(S/'reviews/retained_modules_inventory.json')
config=V/'next_fixed_blocks_20261024/train_parent_audit/frozen_node_configuration.json';frozen=read(config)
structure=read(V/'final_fusion_audit_20261027/structure_audit_agent.json')
rows=[]
for model,p in [('MF587835',N/'broad_official_trials_20261026/readouts/MF'),('Phi587836',N/'broad_official_trials_20261026/readouts/Phi4MM'),('MOSS587634',N/'per_model_diagnostics_20261026/MOSS'),('Omni587632',N/'per_model_diagnostics_20261026/Qwen25Omni7B')]:
 binding=read(p/'binding_manifest.json');fit=read(p/'fit_result.json')
 rows.append(dict(route=model,role='Added point-rating increment around full unchanged PF; not the original PF music expert',readout=ref(p/'fit_result.json'),original_positive_b=fit['positive_b'],normalization='Reuse exact saved six question means/SDs and scalar mean/SD, then original clip[-5,5]; no new estimation',source=ref(p/'binding_manifest.json'),original_raw=binding['axis']['raw'],original_TRAIN_scalar=ref(p/'TRAIN_diagnostic.npz'),original_VALID_CSV=ref(p/'valid_predictions.csv'),report=ref(p/'report.json'),frozen_binding_refs=refs(binding)))
allrefs=refs(inventory['retained_modules'])+refs(inventory['main_engine']['runtime_sources'])+refs(inventory['main_engine']['verified_small_model_assets'])+refs(inventory['feedback_layers']['provenance'])+refs(rows)
allrefs +=[dict(path=k,SHA=v)for k,v in frozen['source_sha256'].items()]
for p in [config,S/'authorization_v3_preserve_all.json',S/'mean4/protocol.json',S/'mean4/technical_binding.json',S/'mean4/TRAIN_binding_agent.json',S/'mean4/TRAIN_reference.npz',S/'joint_adapter_data/manifest.json',D/'original_AB_fresh_checkpoint_index_agent.json',D/'original_AB_parameter_metadata_agent.json',D/'original16_parameter_metadata_agent.json',S/'LMY_fixed_TRAIN_replay/collected/complete.json']:
 allrefs.append(ref(p))
local={};external=[]
for r in allrefs:
 p=Path(r['path'])
 if p.is_file():
  key=str(p.resolve());expected=local.get(key,{}).get('expected_SHA',r['SHA']);assert expected==r['SHA'],('Conflicting frozen references',p)
  if key not in local:
   actual=sha(p);local[key]=dict(path=key,expected_SHA=r['SHA'],actual_SHA=actual,bytes=p.stat().st_size,identity_matched=actual==r['SHA'],copy_policy='Index only in this task; downstream package must preserve bytes and relative mapping')
 else:external.append(dict(**r,local_file_available=False,policy='External index only; no download/copy or arbitrary model substitution'))
assert all(x['identity_matched']for x in local.values())
joint=Path(inventory['joint_feature_model_replay_cache']['file']['path'])
with np.load(joint,allow_pickle=False)as z:
 ti=z['train_signal_ids'].tolist();vi=z['valid_signal_ids'].tolist();tm=z['train_mask'];vm=z['valid_mask'];prompts=z['train_prompts']
 with np.load(V/'final_adaptive_20261027/data/gate_train.npz',allow_pickle=False)as gt,np.load(V/'final_adaptive_20261027/data/gate_valid.npz',allow_pickle=False)as gv:
  checks=dict(TRAIN3072_unique=len(ti)==len(set(ti))==3072,VALID2117_unique=len(vi)==len(set(vi))==2117,TRAIN_exact_ID_order=ti==gt['signal_ids'].tolist(),VALID_exact_ID_order=vi==gv['signal_ids'].tolist(),TRAIN_mask_wordcount=np.array_equal(tm.sum(1),gt['counts']),VALID_mask_wordcount=np.array_equal(vm.sum(1),gv['counts']),TRAIN_prompt_wordcount=np.array_equal(np.asarray([len(p.split())for p in prompts]),gt['counts']),TRAIN_prompt_exact=np.array_equal(prompts,gt['prompts']),TRAIN_severity=np.array_equal(z['train_severity'],gt['severity']),VALID_severity=np.array_equal(z['valid_severity'],gv['severity']),TRAIN_mask_shape=list(tm.shape),VALID_mask_shape=list(vm.shape))
 assert all(v for k,v in checks.items()if not k.endswith('shape')),checks
 modules=[]
 for node in inventory['retained_modules']:
  f=node.get('feature_cache',{}).get('file')
  if f:
   with np.load(f['path'],allow_pickle=False)as a:
    assert a['train_signal_ids'].tolist()==ti and a['valid_signal_ids'].tolist()==vi
   modules.append(dict(node=node['id'],feature=f,TRAIN_VALID_ID_exact=True,shapes=node['feature_cache']['arrays']))
state=read(D/'original16_parameter_metadata_agent.json');ab=read(D/'original_AB_parameter_metadata_agent.json')
headfacts=[]
for r in state['models']:
 actualgroup='MOSS55_original4'if r['group']=='MOSS72_original4'else r['group']
 headfacts.append(dict(group=actualgroup,path=r['path'],SHA=r['actual_SHA'],safe_weights_only_read=r['weights_only'],state_tensors=r['state_tensors']))
nodes=[]
for n in structure['nodes']:
 if n['id']=='PFM':n=dict(n,label='0.81 originalP2 +0.09 originalMF72 +0.10 originalMOSS55; MF72→8→1=593 each×4, MOSS55→8→1=457 each×4; all original checkpoints/ratios retained')
 if n['id']=='jury':n=dict(n,label='Added four user-named frozen point readouts MF/Phi/MOSS/Omni, each original positive OLS b/norm reused; OwnMean4=PF+mean(Pj-PF), original PF counted once')
 if n['id']=='LMY':n=dict(n,label='External latest exported prediction black box, exactVALID3e663; matching conditional/in-sample TRAIN3057 recovered by frozen same-head CPU replay; no latest strict whole-system OOF')
 if n['id']=='joint':n=dict(n,label='Original w0=.5609719773532506 retained as center; only new TRAIN13-parameter sample adapter w=w0+.05*tanh(theta·metadata12), original PF/modules unchanged; formal outcome separately bound by root')
 nodes.append(n)
doc=dict(schema='CLIP2.full_original_PF_plus_mean4.delivery_asset_inventory.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),inventory_only=True,user_scope=ref(S/'authorization_v3_preserve_all.json'),PF_original=inventory['PF_reference'],all_original_PF_modules_models_fields_seeds_coefficients_retained=True,original_node_configuration=ref(config),original_static_values=dict(P2_base=[.28,0,.575,.145],P2_amplification=1.85,music_outer=[.81,.09,.10],first_composition=frozen['first_composition_coefficients'],third_composition=frozen['third_composition_coefficients'],Z13_fixed=frozen['Z13_fixed_coefficient'],joint_original_w0=.5609719773532506),algorithm_nodes=nodes,original_modules=inventory['retained_modules'],original_main16_actual_heads=headfacts,original_AB_actual_heads=ab,original_AB_fresh_external_index=ref(D/'original_AB_fresh_checkpoint_index_agent.json'),added_point_routes=rows,input_ID_mask_words_checks=checks,per_feature_identity=modules,all_local_referenced_SHA_currently_exact=True,local_files=list(local.values()),external_references=external,LMY_external_frozen_80_assets_index=ref(S/'LMY_fixed_TRAIN_replay/manifest_3057.json'),LMY_actual_execution_identity=ref(S/'LMY_fixed_TRAIN_replay/collected/complete.json'),new_adapter_data=ref(S/'joint_adapter_data/manifest.json'),corrections=[dict(fact='Original PF MOSS G02 uses55 input features and457 saved parameters per seed; MF uses72 and593',evidence=ref(D/'original16_parameter_metadata_agent.json'),superseded_text=['Earlier inventory group name MOSS72_original4','Earlier structure node says both music branches72','G02 protocol prose449 parameters'],all_original_model_bytes_preserved=True)],package_policy=dict(new_recipe_is_not_OLDPRIMARY397=True,do_not_copy_old_PRIMARY_as_new_package=True,big_audio_LLM_backbones_external_SHA_index_only=True,no_credentials_or_account_tokens_in_inventory=True,no_large_model_or_feature_copies_performed=True,LMY_internals_not_published_as_our_algorithm=True),remaining=dict(new_joint_official_and_actual_result_binding='Root owns; not decided by this inventory',raw_audio_infer_all='Deferred',strict_nested_whole_pipeline_OOF='Deferred',EVAL_submission='Deferred; not counted as complete'),limitations=['Existing original VALID-informed numeric choices remain accepted baseline under current user instruction. A bounded adapter does not erase historical selection.','TRAIN3057 is conditional/in-sample upstream; only the new adapter can have held diagnostics. No zero-overfit certificate is implied.','The historical source inventory is used for immutable refs/structure only; its superseded deletion/reconstruction recommendations are not adopted.'],operations=dict(new_fit=0,old_CV=0,new_model_forward=0,official_API=0,process_signals=0,large_asset_copies=0))
out=D/'full_original_PF_assets_agent.json'
with out.open('x',encoding='utf-8')as f:json.dump(doc,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# 完整原 PF＋四路 Mean4 交付资产清单','',f'已逐字节重核本地 {len(local)} 个引用文件；原 A 三份 bundle、B 五份 checkpoint 也已在本人4090只读复核 SHA，全部一致。没有复制大骨干、模型前向或训练。','', '原 PF 的 A/B/C/S1、P2、MF/MOSS、全部 Y/ER/Z、Z13 固定外部 prior、原比例与1.85完整保留；ER01→ER02按原替换账目保留，不重复叠加。','', '事实修正：原 PF 的 MF 是72→8→1/593参数，MOSS G02 是55→8→1/457参数。后者旧72/449文字不符合冻结权重实际形状；模型字节未改。C为353、P2gate为211、S1为7101参数/头。','', '四路新增评审使用原6问2耳评分缓存与原正OLS b、原normalizer；Mean4只平均四份完整candidate，PF计一次。LMY作为外部预测黑盒：原VALID3e663与TRAIN3057固定同模型回放已精确绑定，仍不是整链严格OOF。','', '新联合仅学习13参数，以原w0=.5609719773532506为中心逐样本±.05；最终官方状态由root另绑定。本清单不把旧PRIMARY397包视为新配方，也不宣称最终成绩已通过。','', '原raw infer_all、严格嵌套OOF和EVAL提交继续延期。大音频/LLM骨干仅外置SHA索引，不复制凭据或他人资产。','',f'完整机器清单：{out}','']
with (D/'full_original_PF_assets_agent.md').open('x',encoding='utf-8')as f:f.write('\n'.join(lines))
print(json.dumps(dict(complete=True,local_refs=len(local),external_refs=len(external),all_local_SHA_exact=True,inventory=ref(out),fit=0,forward=0,copy=0)))
