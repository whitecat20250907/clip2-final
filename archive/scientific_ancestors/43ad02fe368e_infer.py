from pathlib import Path
import os,json,argparse,time,hashlib
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np,torch
from torch import nn
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,allow_nan=False));t.replace(p)
class Net(nn.Module):
 def __init__(self,dim):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,8),nn.Tanh(),nn.Linear(8,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x,g,m,anchor):
  w=.05*torch.tanh(self.net(torch.cat([x,g[:,None,:].expand(-1,x.shape[1],-1)],-1)).squeeze(-1));return (anchor+(w*m).sum(1)/m.sum(1)).clamp(0,1)
def prepare(t,v,old,fit,arm):
 pre={}
 old=dict(old);old['local_mean']=old['local_mean']+old['music']['mean'];old['local_std']=old['local_std']+old['music']['std']
 for key,prefix in [('local','local'),('global_x','global')]:
  for data in [t,v]:data[key]=data[key].astype(float)*np.array(old[prefix+'_std'])+np.array(old[prefix+'_mean'])
  raw=t[key][fit];raw=raw[t['mask'][fit].astype(bool)] if key=='local' else raw;mu=raw.mean(0);sd=raw.std(0);sd=np.where(sd>1e-6,sd,1);pre[prefix+'_mean']=mu.tolist();pre[prefix+'_std']=sd.tolist()
  for data in [t,v]:data[key]=((data[key]-mu)/sd).astype(np.float32)
 if arm!='base':
  mu=np.zeros(68);sd=np.ones(68)
  for k in range(4):
   good=fit[:,None]&t['mask'].astype(bool)&(t['extra'][:,:,68+k]>0);cols=list(range(16*k,16*(k+1)))+[64+k];raw=t['extra'][good][:,cols];assert len(raw)>100;mu[cols]=raw.mean(0);s=raw.std(0);sd[cols]=np.where(s>1e-5,s,1)
  pre['extra_mean']=mu.tolist();pre['extra_std']=sd.tolist()
  for data in [t,v]:
   a=data['extra'].astype(np.float32).copy()
   for k in range(4):
    good=a[:,:,68+k]>0
    for ix in list(range(16*k,16*(k+1)))+[64+k]:a[:,:,ix]=np.where(good,a[:,:,ix],mu[ix])
   a[:,:,:68]=(a[:,:,:68]-mu)/sd;data['local']=np.concatenate([data['local'],a],-1)
 for data in [t,v]:data['local'][~data['mask'].astype(bool)]=0;assert np.isfinite(data['local']).all()
 return pre
def infer(net,z):
 out=[]
 with torch.inference_mode():
  for lo in range(0,len(z['signal_ids']),256):out.append(net(*[torch.tensor(z[k][lo:lo+256],dtype=torch.float32) for k in ['local','global_x','mask','anchor']]).numpy())
 return np.concatenate(out)

def predict_standardized(local_folds,global_x,mask,anchor):
 b=Path(__file__).resolve().parent;lock=json.loads((b/'manifest.json').read_text());old=json.loads((b/'input_preprocessing.json').read_text());torch.set_num_threads(2)
 local_folds=np.asarray(local_folds);global_x=np.asarray(global_x);mask=np.asarray(mask);anchor=np.asarray(anchor)
 assert local_folds.ndim==4 and local_folds.shape[0]==5 and local_folds.shape[-1]==19
 n,w=mask.shape;assert local_folds.shape[1:3]==(n,w) and global_x.shape==(n,23) and anchor.shape==(n,) and n>0 and (mask.sum(1)>0).all()
 old_mu=np.array(old['local_mean']+old['music']['mean']);old_sd=np.array(old['local_std']+old['music']['std']);parts=[]
 for info in lock['models']:
  p=b/info['file'];assert sha(p)==info['SHA'];ck=torch.load(p,map_location='cpu',weights_only=False);net=Net(ck['dim']);net.load_state_dict(ck['state']);net.eval();pre=ck['preprocessing'];glob=((global_x.astype(float)*np.array(old['global_std'])+np.array(old['global_mean'])-np.array(pre['global_mean']))/np.array(pre['global_std'])).astype(np.float32);folds=[]
  for k in range(5):
   local=((local_folds[k].astype(float)*old_sd+old_mu-np.array(pre['local_mean']))/np.array(pre['local_std'])).astype(np.float32);local[~mask.astype(bool)]=0
   z=dict(signal_ids=np.arange(n),local=local,global_x=glob,mask=mask,anchor=anchor);folds.append(infer(net,z))
  parts.append(np.mean(np.stack(folds),axis=0))
 pred=np.mean(parts,axis=0);assert np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1;return pred
