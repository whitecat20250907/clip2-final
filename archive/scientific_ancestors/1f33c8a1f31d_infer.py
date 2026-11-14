"""Frozen S1 head inference only; upstream raw feature generation and full fusion are separate."""
from pathlib import Path
import json,numpy as np,torch
def model(arm):
 import torch
 from torch import nn
 class Head(nn.Module):
  def __init__(self):
   super().__init__();self.arm=arm;dim={0:100,1:108,2:140}[arm];width={0:69,1:64,2:50}[arm];self.net=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.Linear(width,1))
  def forward(self,l,g,m):
   if self.arm<2:
    x=torch.cat((l[...,:24] if self.arm==0 else l,g[:,None,:].expand(-1,l.shape[1],-1)),-1);q=self.net(x).squeeze(-1).sigmoid();return (q*m).sum(1)/m.sum(1)
   count=m.sum(1)[:,None];mean=(l*m[...,None]).sum(1)/count;variance=((l-mean[:,None,:])**2*m[...,None]).sum(1)/count;return self.net(torch.cat((mean,variance.clamp_min(0).sqrt(),g),-1)).squeeze(-1).sigmoid()
 return Head()

def prepare(raw, directory=None):
 directory=Path(directory or Path(__file__).parent)
 pre=json.loads((directory/'preprocessing.json').read_text(encoding='utf-8'))
 with np.load(directory/'pca.npz',allow_pickle=False) as z: pm=z['mean'];pc=z['components']
 l=np.asarray(raw['local'],dtype=np.float64);m=np.asarray(raw['mask'],dtype=bool)
 assert l.shape[:2]==m.shape and l.shape[2]==32 and np.all(m.sum(1)>0)
 mu=np.asarray(pre['base']['mean']);sd=np.asarray(pre['base']['std'])
 l=(np.where(np.isnan(l),mu,l)-mu)/sd;l[~m]=0
 audio=np.asarray(raw['audio'],dtype=np.float64);sev=np.asarray(raw['severity'],dtype=int);counts=np.asarray(raw['counts'])
 assert audio.shape==(len(l),1280) and np.isin(sev,[0,1,2]).all() and np.array_equal(m.sum(1),counts)
 core=np.column_stack(((audio-pm)@pc.T,np.eye(3)[sev],np.log1p(counts)))
 gm=np.asarray(pre['global_core']['mean']);gs=np.asarray(pre['global_core']['std'])
 g=np.column_stack(((np.where(np.isnan(core),gm,core)-gm)/gs,np.zeros((len(l),40))))
 assert np.isfinite(l).all() and np.isfinite(g).all()
 return l.astype(np.float32),g.astype(np.float32),m

def predict_standardized(l,g,m,directory=None):
 directory=Path(directory or Path(__file__).parent);g=np.asarray(g,dtype=np.float32).copy();g[:,-40:]=0
 values=[]
 for seed in [142,242]:
  net=model(2);state=torch.load(directory/f'seed{seed}.pt',map_location='cpu',weights_only=True);net.load_state_dict(state['state']);net.eval();parts=[]
  with torch.inference_mode():
   for start in range(0,len(l),256):
    sl=slice(start,start+256);parts.append(net(torch.as_tensor(l[sl]),torch.as_tensor(g[sl]),torch.as_tensor(m[sl])).numpy())
  values.append(np.concatenate(parts))
 return .5*(values[0]+values[1])

def predict(raw,directory=None):
 return predict_standardized(*prepare(raw,directory),directory=directory)
