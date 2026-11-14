import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from tdt_loss_core_v1 import streamed_loss_gradient
class ProjectionPair(nn.Module):
 def __init__(self,seed=140044,dtype=torch.float32,device='cuda'):
  super().__init__();torch.manual_seed(seed)
  self.eA=nn.Parameter(torch.randn(8,1024,dtype=dtype,device=device)*.02);self.eB=nn.Parameter(torch.zeros(640,8,dtype=dtype,device=device));self.dA=nn.Parameter(torch.randn(8,640,dtype=dtype,device=device)*.02);self.dB=nn.Parameter(torch.zeros(640,8,dtype=dtype,device=device))
 def acoustic(self,x,e):return e+2*F.linear(F.linear(x,self.eA),self.eB)
 def context(self,d):return d+2*F.linear(F.linear(d,self.dA),self.dB)
 def forward(self,x,e,d):return self.acoustic(x,e),self.context(d)
class AcousticWrapper(nn.Module):
 def __init__(self,base,adapter):super().__init__();self.base=base;self.adapter=adapter
 def forward(self,x):return self.adapter.acoustic(x,self.base(x))
class ContextWrapper(nn.Module):
 def __init__(self,base,adapter):super().__init__();self.base=base;self.adapter=adapter
 def forward(self,x):return self.adapter.context(self.base(x))
def bridge(head,adapter,X,E,D,labels,scale=1.):
 ee,dd=adapter(X,E,D);el=ee.detach().requires_grad_(True);dl=dd.detach().requires_grad_(True)
 loss=streamed_loss_gradient(head,el,dl,labels,scale=scale);assert torch.isfinite(el.grad).all()and torch.isfinite(dl.grad).all()
 torch.autograd.backward((ee,dd),(el.grad,dl.grad));return loss
def qualify_bridge(head,X,E,D,labels):
 from tdt_qualification_v1 import exhaustive_loss
 import copy
 head=copy.deepcopy(head).double().cpu().requires_grad_(False);X=X[:3].double().cpu();E=E[:3].double().cpu();labels=labels[:2];D=D[:len(labels)+1].double().cpu();ad=ProjectionPair(dtype=torch.float64,device='cpu')
 with torch.no_grad():ad.eB.normal_(std=.001);ad.dB.normal_(std=.001)
 ee,dd=ad(X,E,D);loss=exhaustive_loss(head,ee,dd,labels);loss.backward();g=[v.grad.clone()for v in ad.parameters()];ad.zero_grad(set_to_none=True);stream=bridge(head,ad,X,E,D,labels);gap=max(float((v.grad-q).abs().max())for v,q in zip(ad.parameters(),g));lg=abs(float(loss.detach())-stream);assert gap<1e-8 and lg<1e-8
 return dict(full_token_vocabulary=8193,T=len(X),L=len(labels),loss_gap=lg,adapter_gradient_maxgap=gap,proof=True)
