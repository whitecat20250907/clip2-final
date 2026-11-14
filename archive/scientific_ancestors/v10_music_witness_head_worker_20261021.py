"""Independent frozen-music readout or response-only correction, matched controls."""
from pathlib import Path
import os,json,argparse,time
import numpy as np,torch
def save(path,z):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(z));tmp.replace(path)
def arr(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
class Net(torch.nn.Module):
    def __init__(self,dim,arm):
        super().__init__();self.arm=arm;self.head=torch.nn.Sequential(torch.nn.Linear(dim,8),torch.nn.Tanh(),torch.nn.Linear(8,1));torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,mask,parent):
        word=self.head(x).squeeze(-1)
        if self.arm=='witness':return (torch.sigmoid(word)*mask).sum(1)/mask.sum(1)
        delta=.05*(torch.tanh(word)*mask).sum(1)/mask.sum(1);return torch.clamp(parent+delta,0,1)
def infer(net,z):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(z['signal_ids']),256):
            sl=slice(start,start+256);result.append(net(*(torch.tensor(z[k][sl],device='cuda',dtype=torch.float32) for k in ['x','mask','parent'])).cpu().numpy())
    return np.concatenate(result)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert os.getuid()==b.stat().st_uid==p['owner_uid'] and a.role in p['roles'];arm,ff,ss=a.role.split('_');f=-1 if ff=='full' else int(ff[1:]);seed=int(ss[1:]);assert arm in ['witness','residual','perm'];d=b/'tasks'/a.role
    t=arr(b/'train.npz');v=arr(b/'valid.npz');assert len(t['signal_ids'])==p.get('expected_train_rows',3072) and len(v['signal_ids'])==2117 and 'target' not in v
    fit=np.ones(len(t['signal_ids']),bool) if f<0 else t['fold']!=f;hold=~fit
    if f>=0:assert not set(t['prompts'][fit])&set(t['prompts'][hold])
    raw=t['x'][t['permutation']] if arm=='perm' else t['x'];values=raw[fit][t['mask'][fit].astype(bool)];mu=values.mean(0);sd=values.std(0);sd=np.where(sd>1e-6,sd,1);t['x']=((raw-mu)/sd).astype(np.float32);v['x']=((v['x']-mu)/sd).astype(np.float32)
    t['x'][~t['mask'].astype(bool)]=0;v['x'][~v['mask'].astype(bool)]=0
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.07);torch.manual_seed(seed);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    net=Net(t['x'].shape[-1],arm).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);xx=[torch.tensor(t[k][fit],device='cuda',dtype=torch.float32) for k in ['x','mask','parent']];target=torch.tensor(t['target'][fit],device='cuda',dtype=torch.float32);rng=np.random.default_rng(seed);start=time.monotonic()
    for epoch in range(80):
        for ix in np.array_split(rng.permutation(len(target)),int(np.ceil(len(target)/128))):
            pred=net(*(q[ix] for q in xx));loss=(pred-target[ix]).square().mean()
            if arm!='witness':loss=loss+.1*(pred-xx[2][ix]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if epoch%10==9:save(d/'progress.json',dict(epoch=epoch+1,seconds=time.monotonic()-start))
    net.eval();vp=infer(net,v);np.savez_compressed(d/'valid.npz',signal_ids=v['signal_ids'],prediction=vp)
    if f>=0:
        hh={k:x[hold] for k,x in t.items() if k!='permutation'};pred=infer(net,hh);np.savez_compressed(d/'held.npz',signal_ids=hh['signal_ids'],prediction=pred,target=hh['target'])
    torch.save(dict(state=net.cpu().state_dict(),arm=arm,fold=f,seed=seed,mean=mu,std=sd,dim=t['x'].shape[-1]),d/'model.pt');save(d/'result.json',dict(status='frozen_music_readout_complete',arm=arm,fold=f,seed=seed,parameters=sum(x.numel() for x in net.parameters()),seconds=time.monotonic()-start,large_model_trained=False,VALID_EVAL_human_targets_read=False,conditional_not_joint_nested=True))
if __name__=='__main__':main()
