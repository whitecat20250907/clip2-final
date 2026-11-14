"""C353 objective-only intervention with independently frozen outer-fold gates."""
from pathlib import Path
import os,sys,json,hashlib,time,argparse,importlib.util
import numpy as np,torch
from v10_stall_frozen_fusion_20261021 import FrozenFusion
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,z):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(z,indent=2));q.replace(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--base');a.add_argument('--role');a=a.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==p['owner_uid'] and a.role in p['roles']
 assert sha(__file__)==p['worker_SHA'] and sha(b/'v10_stall_frozen_fusion_20261021.py')==p['fusion_SHA']
 for n,d in p['files'].items():assert sha(b/n)==d,n
 arm,fs,ss=a.role.split('_');f=-1 if fs=='full' else int(fs[1:]);seed=int(ss[1:]);assert arm=='base';dest=b/'tasks'/a.role
 spec=importlib.util.spec_from_file_location('original_C353',b/'original_worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 t=arrays(b/'train_word.npz');v=arrays(b/'valid_word.npz');env=arrays(b/'train_environment.npz');assert np.array_equal(t['signal_ids'],env['signal_ids']) and len(t['signal_ids'])==16047
 fit=np.ones(len(t['signal_ids']),bool) if f<0 else t['fold']!=f;held=~fit
 if f>=0:assert not set(t['prompts'][fit])&set(t['prompts'][held])
 pre=m.prepare(t,v,json.loads((b/'preprocessing.json').read_bytes()),fit,'base');torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.06);torch.backends.cuda.matmul.allow_tf32=False
 net=m.Net(t['local'].shape[-1]+t['global_x'].shape[-1]).cuda()
 gate_names=[('gate_full_' if f<0 else 'gate_f'+str(f)+'_')+str(s)+'.pt' for s in [142,242]]
 gates=[torch.load(b/n,map_location='cpu',weights_only=False) for n in gate_names];fusion=FrozenFusion(gates).cuda().eval();assert not any(q.requires_grad for q in fusion.parameters())
 x=[torch.tensor(t[k][fit],dtype=torch.float32,device='cuda') for k in ['local','global_x','mask','anchor']];y=torch.tensor(t['target'][fit],dtype=torch.float32,device='cuda')
 en={k:torch.as_tensor(env[k][fit],device='cuda',dtype=torch.float32) for k in ['prediction','audio8','severity','counts']}
 assert torch.equal(en['prediction'][:,0],torch.as_tensor(t['A_OOF'][fit],device='cuda')) and torch.equal(en['prediction'][:,3],torch.as_tensor(t['S1_full_OOF'][fit],device='cuda'))
 opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);rng=np.random.default_rng(seed);start=time.monotonic()
 for epoch in range(80):
  assert time.monotonic()-start<1800
  for ix in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/128))):
   cp=net(*(q[ix] for q in x));score=fusion(cp,{k:q[ix] for k,q in en.items()});loss=(score-y[ix]).square().mean()+.1*(cp-x[3][ix]).square().mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();assert all(q.grad is None for q in fusion.parameters());opt.step()
  save(dest/'progress.json',dict(epoch=epoch+1,seconds=time.monotonic()-start,objective='frozen_final_dynamic_fusion'))
 ck=dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},arm='base',dim=t['local'].shape[-1]+t['global_x'].shape[-1],preprocessing=pre,fold=f,seed=seed,protocol_SHA=sha(b/'protocol.json'),objective='frozen_final_dynamic_fusion',gate_SHAs={n:sha(b/n) for n in gate_names})
 torch.save(ck,dest/'model.pt');net=net.cpu().eval();valids=[v];prior=json.loads((b/'preprocessing.json').read_bytes());prior['local_mean']+=prior['music']['mean'];prior['local_std']+=prior['music']['std']
 for vf in range(1,5):
  zz=arrays(b/f'valid_fold{vf}.npz');assert np.array_equal(zz['signal_ids'],v['signal_ids']) and 'target' not in zz
  for key,prefix in [('local','local'),('global_x','global')]:zz[key]=((zz[key].astype(float)*np.array(prior[prefix+'_std'])+np.array(prior[prefix+'_mean'])-np.array(pre[prefix+'_mean']))/np.array(pre[prefix+'_std'])).astype(np.float32)
  zz['local'][~zz['mask'].astype(bool)]=0;valids.append(zz)
 pred=np.mean([m.infer(net,z) for z in valids],0);re=m.Net(ck['dim']);re.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=False)['state']);re.eval();again=np.mean([m.infer(re,z) for z in valids],0);delta=float(np.max(abs(pred-again)));assert delta<=1e-6
 np.savez_compressed(dest/'valid.npz',signal_ids=v['signal_ids'],prediction=pred)
 if f>=0:
  hh={k:v[held] for k,v in t.items()};hp=m.infer(net,hh);np.savez_compressed(dest/'held.npz',signal_ids=hh['signal_ids'],prediction=hp,target=hh['target'])
 save(dest/'result.json',dict(status='objective_only_C353_complete',fold=f,seed=seed,seconds=time.monotonic()-start,model_SHA=sha(dest/'model.pt'),CPU_reload_difference=delta,gate_SHAs=ck['gate_SHAs'],gate_parameters_frozen=True,C_dependency_through_weights_preserved=True,conditional_not_joint_nested=True,VALID_targets_read=False,historical_gate_fit_contains_8_additionally_removed_rows=True))
if __name__=='__main__':main()
