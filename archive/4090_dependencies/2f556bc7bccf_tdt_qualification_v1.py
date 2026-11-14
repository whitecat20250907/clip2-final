import numpy as np,torch
from torch.nn import functional as F
from tdt_loss_core_v1 import LowRankJoint,streamed_loss_gradient,tdt_expected
def exhaustive_loss(head,E,D,labels):
 T,U=len(E),len(D);z=head(F.relu(E[:,None,:]+D[None,:,:]));V=z.shape[-1]-5;lp=z[:,:,:V].log_softmax(-1);du=z[:,:,V:].log_softmax(-1);paths=[]
 def visit(t,u,v):
  if t==T:
   if u==len(labels):paths.append(v)
   return
  for d in range(1,5):
   if t+d<=T:visit(t+d,u,v+lp[t,u,V-1]+du[t,u,d])
  if u<len(labels):
   for d in range(5):
    if t+d<T:visit(t+d,u+1,v+lp[t,u,labels[u]]+du[t,u,d])
 visit(0,0,z.new_tensor(0.));assert paths;return -torch.logsumexp(torch.stack(paths),0)/(T+len(labels))
def qualify(actual=None):
 out=[];rng=np.random.default_rng(41)
 cases=[(1,[]),(1,[0]),(2,[0,0]),(3,[0,1])]
 for index,(T,labels)in enumerate(cases):
  W=torch.tensor(rng.normal(size=(9,7))*.1,dtype=torch.float64);b=torch.tensor(rng.normal(size=9)*.1,dtype=torch.float64);E=torch.tensor(rng.normal(size=(T,7)),dtype=torch.float64);D=torch.tensor(rng.normal(size=(len(labels)+1,7)),dtype=torch.float64);head=LowRankJoint(W,b,700+index)
  with torch.no_grad():head.B.normal_(std=.01)
  loss=exhaustive_loss(head,E,D,labels);loss.backward();g=[v.grad.clone()for v in [head.A,head.B]];head.zero_grad(set_to_none=True);stream=streamed_loss_gradient(head,E,D,labels);gap=max(float((v.grad-q).abs().max())for v,q in zip([head.A,head.B],g));lgap=abs(float(loss.detach())-stream);assert gap<1e-8 and lgap<1e-8;out.append(dict(case=index,T=T,L=len(labels),loss_gap=lgap,LoRA_gradient_maxgap=gap))
 if actual is not None:
  W,b,E,D,labels=actual;head=LowRankJoint(W.double().cpu(),b.double().cpu(),991)
  with torch.no_grad():head.B.normal_(std=.001)
  E=E[:3].double().cpu();labels=labels[:2];D=D[:len(labels)+1].double().cpu();loss=exhaustive_loss(head,E,D,labels);loss.backward();g=[v.grad.clone()for v in [head.A,head.B]];head.zero_grad(set_to_none=True);stream=streamed_loss_gradient(head,E,D,labels);gap=max(float((v.grad-q).abs().max())for v,q in zip([head.A,head.B],g));lgap=abs(float(loss.detach())-stream);assert gap<1e-8 and lgap<1e-8;out.append(dict(case='actual_FIT_FULL8193_cropped',T=len(E),L=len(labels),loss_gap=lgap,LoRA_gradient_maxgap=gap))
 return out
