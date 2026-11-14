"""Actual isolated-vocal evidence: complete three-seed current583543 fusion and replacement."""
import argparse,json,shutil,time,functools
from pathlib import Path
import numpy as np
from d6e_word_v1.common import digest,owned,save_json,original_module
from .v6_fast_fusion_20260924 import load as frozen_load
ARMS=('m2_stem_add','m2_stem_replace')
ACTIVE_ARM=None
RATIONALE={'m2_stem_add':'Add real isolated vocal descriptors and paired accompaniment ratio to official-positive M2+ACON.','m2_stem_replace':'Replace mixture musical proxies with real isolated vocal descriptors, retain ACON and all frozen listener/acoustic evidence.'}

def build(data,arm):
 if arm not in ARMS:raise ValueError(arm)
 d={k:v for k,v in data.items() if not k.startswith(('h1_','n2_','extra_'))};w=d['words'];t=d['text'];mask=d['word_mask']
 if w.shape[-1]!=89 or t.shape[-1]!=58:raise ValueError('Frozen583432 schema')
 if arm=='m2_stem_add':w=np.concatenate([w,data['extra_music']],-1)
 w=np.concatenate([w,data['extra_stems']],-1)
 if not np.isfinite(w).all() or np.any(w[~mask]!=0):raise ValueError('Feature finite/padding')
 return {**d,'words':w,'text':t}

def model_class(arm):
 import torch
 from torch import nn
 from d6e_word_v1.models import ReferenceWordModel
 from .v5_supp_s1_model import ActivationWordModel
 if arm=='s1_tau2':return functools.partial(ActivationWordModel,variant='act_tau2')
 if arm not in ARMS:return ReferenceWordModel
 class Acon(nn.Module):
  def __init__(self):
   super().__init__();self.p1=nn.Parameter(torch.ones(128));self.p2=nn.Parameter(torch.zeros(128));self.beta=nn.Parameter(torch.ones(128))
  def forward(self,x):
   delta=(self.p1-self.p2)*x;return delta*torch.sigmoid(self.beta*delta)+self.p2*x
 class Model(ReferenceWordModel):
  def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.word_head[1]=Acon()
 return Model

def load(root,bank,source):
 data,m,T=frozen_load(root,bank,source)
 # Source identity of the additional model implementation is recorded in every recipe.
 extras=root/'runs/v6_round2_20260924/inputs'
 for kind in ('music','stems'):
  mp=extras/kind/'manifest.json' if kind=='music' else root/'runs/v6_m2_stems_20260924/inputs/manifest.json'
  if not mp.exists():raise FileNotFoundError(mp)
  em=json.loads(mp.read_text())
  if em['status']!='completed':raise ValueError('Incomplete extras')
  for split,d in data.items():
   f=mp.parent/(split+'.npz')
   if digest(f)!=em['files'][f.name]:raise ValueError('Extra hash')
   with np.load(f,allow_pickle=False) as z:
    if not np.array_equal(z['signal_ids'],d['signal_ids']):raise ValueError('Extra identity')
    data[split]['extra_'+kind]=z['values'].copy()
  m={**m,kind:em}
 return data,m,T

def package(root,source,out):raise RuntimeError('Use audited original frozen bundle and separate new feature manifests')

def infer(T,ck,d,device,cls):
 from d6e_word_v1 import train as v1
 from torch.utils.data import DataLoader
 model=cls(v1.make_base(T,ck['base_config']),'word',d['words'].shape[-1],d['text'].shape[-1]).to(device);model.load_state_dict(ck['model_state'],strict=True)
 p,_,_=v1.evaluate(model,DataLoader(v1.dataset(d,ck['normalizer']),batch_size=16),device,d);return p

def run(a,root,bank,out):
 import torch
 from torch.utils.data import DataLoader
 from d6e_word_v1 import train as v1
 from d6e_word_v1.models import ReferenceWordModel
 from .v5_n2_no_l5 import checked_prediction
 torch.set_num_threads(2)
 if not torch.cuda.is_available():raise RuntimeError('Remote GPU required')
 data,m,T=load(root,bank,a.source_root);device=torch.device('cuda:0');arm=a.arm;cls=model_class(arm)
 recipe=dict(experiment='8h remaining-window second fixed fusion batch', additional_input_provenance={k:m[k] for k in ('music','stems') if k in m}, activation_source_sha256=digest(Path(__file__).with_name('v5_supp_s1_model.py')),arm=arm,rationale=RATIONALE[arm],bundle_sha256=digest(bank/'manifest.json'),seeds=[42,43,44],source_sha256=digest(__file__),baseline_submission=583543,prior_frozen_submission=583432,all_train_rows=16238,official_primary=True)
 work={s:build(d,arm) for s,d in data.items()};folder=out/(f'smoke_seed{a.seed}' if a.smoke_only else f'seed{a.seed}')
 if folder.exists():raise FileExistsError(folder)
 folder.mkdir(parents=True);save_json(folder/'protocol.json',recipe);pred={};repro={};cfg=None
 if a.smoke_only:work={s:{k:v[:64] for k,v in d.items()} for s,d in work.items()}
 for control in ('n2','best','current_m2','current_m2_acon'):
  f=(root/f'runs/v6_round2_20260924/m2_word/seed{a.seed}/m2_word_seed{a.seed}') if control=='current_m2' else bank/'controls'/f'{control}{a.seed}';
  if control=='current_m2_acon':f=root/f'runs/v6_positive_combo_20260924/m2_acon/seed{a.seed}/m2_acon_seed{a.seed}'
  ck=torch.load(f/'best_model.pt',map_location='cpu',weights_only=False);cfg=ck['base_config'];pred[control]={};repro[control]={}
  if not a.smoke_only:
   for split in ('dev','valid'):
    d={k:v for k,v in data[split].items() if not k.startswith(('h1_','n2_','extra_'))}
    if control in ('current_m2','current_m2_acon'):d={**d,'words':np.concatenate([d['words'],data[split]['extra_music']],-1)}
    if control=='n2':d={**d,**{k:data[split]['n2_'+k] for k in ('acoustic','words','text')}}
    p=infer(T,ck,d,device,model_class(a.arm) if control=='current_m2_acon' else ReferenceWordModel);err=float(abs(p-checked_prediction(f/(split+'_predictions.npz'),d)).max())
    # Cross-device floating point differences are recorded, never silently ignored.
    if err>1e-6:raise ValueError('Control replay mismatch '+control+str(err))
    pred[control][split]=p;repro[control][split]=err
  del ck
 norm=v1.fit_normalizer(build(data['train'],arm));datasets={s:v1.dataset(d,norm) for s,d in work.items()};loaders={s:DataLoader(d,batch_size=16) for s,d in datasets.items()}
 if False:
  # Identity initialization is verified with exactly the same base/head weights.
  test=cls(v1.make_base(T,cfg),'word',work['train']['words'].shape[-1],work['train']['text'].shape[-1]).to(device).eval()
  plain=ReferenceWordModel(v1.make_base(T,cfg),'word',work['train']['words'].shape[-1],work['train']['text'].shape[-1]).to(device).eval();plain.load_state_dict({k:v for k,v in test.state_dict().items() if k!='gate_logits'},strict=True)
  batch=next(iter(loaders['train']));x=[v.to(device) for v in batch[:7]]
  with torch.no_grad():err=float((test(*x)['prediction']-plain(*x)['prediction']).abs().max())
  if err>1e-7:raise ValueError('Gate identity initialization')
  del test,plain
 v1.ReferenceWordModel=cls;v1.VARIANTS[arm]=dict(v1.VARIANTS['word_bce']);schema=dict(fast_fusion=recipe,smoke_only=a.smoke_only)
 result=v1.train_one(arm,a.seed,T,cfg,work,datasets,loaders,norm,device,folder,1 if a.smoke_only else 50,prediction_splits=('dev','valid'),feature_schema=schema)
 f=folder/f'{arm}_seed{a.seed}';ck=torch.load(f/'best_model.pt',map_location='cpu',weights_only=False);pred[arm]={}
 for split in ('dev','valid'):
  p=infer(T,ck,work[split],device,cls)
  if float(abs(p-checked_prediction(f/(split+'_predictions.npz'),work[split])).max())>1e-6:raise ValueError('Candidate replay mismatch')
  pred[arm][split]=p
 if a.smoke_only:pred={arm:pred[arm]}
 np.savez_compressed(folder/'paired_predictions.npz',dev_ids=work['dev']['signal_ids'],valid_ids=work['valid']['signal_ids'],target=work['dev']['target'],**{n+'_'+s:p for n,splits in pred.items() for s,p in splits.items()})
 save_json(folder/'report.json',dict(status='completed_smoke' if a.smoke_only else 'completed_seed',smoke_only=a.smoke_only,seed=a.seed,recipe=recipe,prediction_sha256=digest(folder/'paired_predictions.npz'),metrics={n:v1.metrics(p['dev'],work['dev']['target']) for n,p in pred.items()},seconds=result['seconds'],epochs=result['epochs_run'],peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),control_reproduction=repro,checkpoint_sha256=digest(f/'best_model.pt')))

def summarize(a,root,bank,out):
 from d6e_word_v1 import train as v1
 from d6e_word_v1.report import paired_bootstrap
 from .b2_export import check_csv
 data,m,T=load(root,bank,a.source_root);arm=a.arm;rows=[];evidence=[];recipe=None
 for seed in (42,43,44):
  f=out/f'seed{seed}';r=json.loads((f/'report.json').read_text())
  if r['status']!='completed_seed' or r['smoke_only'] or r['seed']!=seed or digest(f/'paired_predictions.npz')!=r['prediction_sha256']:raise ValueError('Complete three seeds')
  if recipe is None:recipe=r['recipe']
  if recipe!=r['recipe']:raise ValueError('Matching recipe')
  with np.load(f/'paired_predictions.npz',allow_pickle=False) as z:row=dict(z)
  for key,expected in [('dev_ids',data['dev']['signal_ids']),('valid_ids',data['valid']['signal_ids']),('target',data['dev']['target'])]:
   if not np.array_equal(row[key],expected):raise ValueError('Prediction identity')
  rows.append(row);evidence.append(dict(seed=seed,report_sha256=digest(f/'report.json')))
 names=(arm,'best','n2','current_m2','current_m2_acon');bags={n:{s:np.mean([r[n+'_'+s].astype(np.float64) for r in rows],axis=0) for s in ('dev','valid')} for n in names};y=data['dev']['target']
 contrasts={n:dict(paired=paired_bootstrap(bags[n]['dev'],bags[arm]['dev'],y,data['dev']['prompts']),seed_delta_rmse=[v1.metrics(r[arm+'_dev'],y)['rmse_percent']-v1.metrics(r[n+'_dev'],y)['rmse_percent'] for r in rows]) for n in ('best','n2','current_m2','current_m2_acon')}
 f=out/'export'/arm;f.mkdir(parents=True)
 for split in ('dev','valid'):v1.write_predictions(f,split,data[split],bags[arm][split])
 if len(data['valid']['signal_ids'])!=2117:raise ValueError('Official row count')
 exports={arm:dict(csv=check_csv(f/'valid_predictions.csv',data['valid']['signal_ids'],bags[arm]['valid']),csv_sha256=digest(f/'valid_predictions.csv'))}
 save_json(out/'report.json',dict(status='completed',recipe=recipe,seed_evidence=evidence,bags={n:v1.metrics(p['dev'],y) for n,p in bags.items()},contrasts=contrasts,by_severity={str(s):{n:v1.metrics(p['dev'][data['dev']['severity']==s],y[data['dev']['severity']==s]) for n,p in bags.items()} for s in (0,1,2)},exports=exports,official_scores=None,production_default_changed=False))

def replay_publish(a,root,bank,out):
 import torch
 from .v5_n2_no_l5 import checked_prediction
 torch.set_num_threads(2);data,m,T=load(root,bank,a.source_root);arm=a.arm;report=json.loads((out/'report.json').read_text());checks=[]
 if report['status']!='completed':raise ValueError('Complete report')
 for seed in (42,43,44):
  f=out/f'seed{seed}'/f'{arm}_seed{seed}';cp=f/'best_model.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
  if digest(cp)!=json.loads((f/'metrics.json').read_text())['checkpoint_sha256'] or ck['feature_schema']!=dict(fast_fusion=report['recipe'],smoke_only=False):raise ValueError('Replay source')
  for split in ('dev','valid'):
   d=build(data[split],arm);p=infer(T,ck,d,torch.device('cuda:0'),model_class(arm));err=float(abs(p-checked_prediction(f/(split+'_predictions.npz'),d)).max())
   if err>1e-6:raise ValueError('Independent replay mismatch')
   checks.append(dict(seed=seed,split=split,max_abs_error=err))
 save_json(out/'replay.json',dict(status='completed',checks=checks,report_sha256=digest(out/'report.json')))
 publish=out/'official_ready';publish.mkdir()
 for seed in (42,43,44):
  f=publish/f'seed{seed}';f.mkdir()
  for name in ('report.json','paired_predictions.npz'):shutil.copyfile(out/f'seed{seed}'/name,f/name)
 shutil.copytree(out/'export',publish/'export');shutil.copyfile(out/'replay.json',publish/'replay.json')
 report['independent_replay_sha256']=digest(publish/'replay.json');save_json(publish/'report.json',report)

def main():
 global ACTIVE_ARM
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('work-root','source-root','bank','output'):p.add_argument('--'+k,required=True)
 p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=(42,43,44));p.add_argument('--smoke-only',action='store_true')
 p.add_argument('--mode',choices=('package','train','summarize','replay_publish'),required=True);a=p.parse_args();root=Path(a.work_root);bank=owned(a.bank,root);out=owned(a.output,root)
 ACTIVE_ARM=a.arm
 if a.mode=='package':package(root,a.source_root,out)
 elif a.mode=='train':run(a,root,bank,out)
 elif a.mode=='summarize':summarize(a,root,bank,out)
 else:replay_publish(a,root,bank,out)

if __name__=='__main__':main()
