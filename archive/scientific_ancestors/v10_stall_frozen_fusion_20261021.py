"""Differentiable equivalent of the existing frozen 9.25pp fusion policy."""
import torch
from torch import nn
class Gate(nn.Module):
 def __init__(self,dim=22):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,8),nn.Tanh(),nn.Linear(8,3))
 def forward(self,x,limit=.05):
  raw=self.net(x).tanh();a=raw[:,0];b=raw[:,1].clamp_min(0);s=raw[:,2];c=-(a+b+s)
  d=torch.stack((a,b,c,s),1);d=d/torch.maximum(torch.ones_like(d[:,:1]),d.abs().amax(1,keepdim=True))
  w=torch.tensor([.28,0,.575,.145],device=x.device)+limit*d
  return torch.where((x.abs()<=6).all(1)[:,None],w,torch.tensor([.28,0,.575,.145],device=x.device))
class FrozenFusion(nn.Module):
 def __init__(self,checkpoints):
  super().__init__();self.gates=nn.ModuleList()
  for i,ck in enumerate(checkpoints):
   g=Gate();g.load_state_dict(ck['state']);g.eval().requires_grad_(False);self.gates.append(g)
   self.register_buffer('mean'+str(i),torch.as_tensor(ck['mean'],dtype=torch.float32));self.register_buffer('std'+str(i),torch.as_tensor(ck['std'],dtype=torch.float32))
 def forward(self,c,env,return_weights=False):
  parent=env['prediction'];p=torch.stack((parent[:,0],parent[:,1],c,parent[:,3]),1)
  pairs=torch.stack([p[:,i]-p[:,j] for i in range(4) for j in range(i+1,4)],1)
  x=torch.cat((env['audio8'],torch.nn.functional.one_hot(env['severity'].long(),3).float(),torch.log1p(env['counts'].float())[:,None],p,pairs),1)
  parts=[g((x-getattr(self,'mean'+str(i)))/getattr(self,'std'+str(i))).double() for i,g in enumerate(self.gates)]
  w0=torch.tensor([.28,0,.575,.145],device=c.device,dtype=torch.float64);w=w0+1.85*(torch.stack(parts).mean(0)-w0)
  w=torch.stack((w[:,0],w[:,1],1-w[:,[0,1,3]].sum(1),w[:,3]),1)
  result=(p.double()*w).sum(1)
  return (result,w) if return_weights else result
