"""Frozen baseline plus a tiny group-regularized logistic correction."""
from pathlib import Path
import os,json,argparse,time
import numpy as np,torch

def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def design(x,mask,severity,mu,sd):
    v=(x-mu)/sd;v[~mask.astype(bool)]=0
    selected=[0,1,2,3,8,9,10,11,32,33];out=[np.ones((*x.shape[:2],1),np.float32),v,np.broadcast_to((severity==1)[:,None,None],(*x.shape[:2],1)),np.broadcast_to((severity==2)[:,None,None],(*x.shape[:2],1)),v[:,:,selected]*(severity==1)[:,None,None],v[:,:,selected]*(severity==2)[:,None,None]]
    return np.concatenate(out,2).astype(np.float32)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--role',required=True);a=p.parse_args();b=Path(a.base);cfg=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==cfg['owner_uid'] and a.role in cfg['roles'];arm,seed=a.role.split('_s');seed=int(seed);assert arm in ['true','shuffle'];d=b/'tasks'/a.role;d.mkdir(parents=True,exist_ok=True);t=arrays(b/'train.npz');v=arrays(b/'valid.npz');assert len(t['signal_ids'])==3072 and len(v['signal_ids'])==2117 and 'target' not in v;key='x' if arm=='true' else 'shuffled';mask=t['mask'].astype(bool);mu=t[key][mask].mean(0);sd=t[key][mask].std(0);sd=np.where(sd>1e-6,sd,1);x=design(t[key],t['mask'],t['severity'],mu,sd);vx=design(v[key],v['mask'],v['severity'],mu,sd);dim=x.shape[-1]
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.06);torch.backends.cuda.matmul.allow_tf32=False
    net=torch.nn.Linear(dim,1,bias=False,device='cuda');torch.nn.init.zeros_(net.weight);xx=torch.tensor(x,device='cuda');yy=torch.tensor(t['target'],device='cuda');mm=torch.tensor(t['mask'],device='cuda');parent=torch.tensor(t['parent'],device='cuda');anchor=torch.logit(parent.clamp(1e-6,1-1e-6));penalty=torch.ones(dim,device='cuda')*.01;penalty[39:]=.05;penalty[0]=.001;opt=torch.optim.Adam(net.parameters(),lr=.001);rng=np.random.default_rng(seed);start=time.monotonic()
    def predict(z,m,base):
        shift=.2*torch.tanh(net(z).squeeze(-1));shift=(shift*m).sum(1)/m.sum(1);return torch.sigmoid(torch.logit(base.clamp(1e-6,1-1e-6))+shift)
    with torch.inference_mode():zero=predict(torch.tensor(vx,device='cuda'),torch.tensor(v['mask'],device='cuda'),torch.tensor(v['parent'],device='cuda')).cpu().numpy();gap=float(np.max(np.abs(zero-v['parent'])));assert gap<2e-7
    for epoch in range(80):
        for ix in np.array_split(rng.permutation(len(yy)),24):
            pred=predict(xx[ix],mm[ix],parent[ix]);loss=(pred-yy[ix]).square().mean()+(net.weight.squeeze()*net.weight.squeeze()*penalty).sum();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if epoch%10==9:(d/'progress.json').write_text(json.dumps(dict(epoch=epoch+1,seconds=time.monotonic()-start)))
    with torch.inference_mode():pred=predict(torch.tensor(vx,device='cuda'),torch.tensor(v['mask'],device='cuda'),torch.tensor(v['parent'],device='cuda')).cpu().numpy()
    change=float(np.max(np.abs(pred-v['parent'])));assert np.isfinite(pred).all() and change<=.050001;np.savez_compressed(d/'valid.npz',signal_ids=v['signal_ids'],prediction=pred);torch.save(dict(weight=net.weight.detach().cpu(),mean=mu,std=sd,selected_interactions=[0,1,2,3,8,9,10,11,32,33],seed=seed),d/'model.pt');(d/'complete.json').write_text(json.dumps(dict(status='tiny_relation_correction_complete',arm=arm,seed=seed,parameters=dim,zero_replay_max_abs=gap,max_abs_correction=change,seconds=time.monotonic()-start,TRAIN_performance_gate=False,VALID_targets_read=False,frozen_previous_heads=True)))
