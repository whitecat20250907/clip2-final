import math,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from numba import njit
@njit(cache=True)
def ladd(a,b):
 if a==-np.inf:return b
 if b==-np.inf:return a
 m=max(a,b);return m+math.log1p(math.exp(-abs(a-b)))
@njit(cache=True)
def tdt_expected(blank,label,duration):
 T,U=blank.shape;L=U-1;alpha=np.full((T+1,U),-np.inf);alpha[0,0]=0.
 for t in range(T):
  for u in range(U):
   for d in range(1,5):
    if t+d<=T:alpha[t+d,u]=ladd(alpha[t+d,u],alpha[t,u]+blank[t,u]+duration[t,u,d])
   if u<L:
    for d in range(5):
     if t+d<T:alpha[t+d,u+1]=ladd(alpha[t+d,u+1],alpha[t,u]+label[t,u]+duration[t,u,d])
 Z=alpha[T,L];beta=np.full((T+1,U),-np.inf);beta[T,L]=0.
 for t in range(T-1,-1,-1):
  for u in range(U-1,-1,-1):
   v=-np.inf
   for d in range(1,5):
    if t+d<=T:v=ladd(v,blank[t,u]+duration[t,u,d]+beta[t+d,u])
   if u<L:
    for d in range(5):
     if t+d<T:v=ladd(v,label[t,u]+duration[t,u,d]+beta[t+d,u+1])
   beta[t,u]=v
 gb=np.zeros((T,U));gy=np.zeros((T,U));gd=np.zeros((T,U,5));norm=T+L
 for t in range(T):
  for u in range(U):
   for d in range(1,5):
    if t+d<=T:
     v=math.exp(alpha[t,u]+blank[t,u]+duration[t,u,d]+beta[t+d,u]-Z)/norm;gb[t,u]-=v;gd[t,u,d]-=v
   if u<L:
    for d in range(5):
     if t+d<T:
      v=math.exp(alpha[t,u]+label[t,u]+duration[t,u,d]+beta[t+d,u+1]-Z)/norm;gy[t,u]-=v;gd[t,u,d]-=v
 return -Z/norm,gb,gy,gd,abs(Z-beta[0,0])
class LowRankJoint(nn.Module):
 def __init__(self,W,b,seed):
  super().__init__();self.register_buffer('W',W.detach().clone());self.register_buffer('b',b.detach().clone());torch.manual_seed(seed);self.A=nn.Parameter(torch.randn(8,W.shape[1],dtype=W.dtype,device=W.device)*.02);self.B=nn.Parameter(torch.zeros(W.shape[0],8,dtype=W.dtype,device=W.device))
 def forward(self,x):return F.linear(x,self.W,self.b)+2*F.linear(F.linear(x,self.A),self.B)
def selected_lp(head,E,D,labels,start,end):
 U=len(labels)+1;ii=torch.arange(start,end,device=E.device);tt=torch.div(ii,U,rounding_mode='floor');uu=ii%U;x=F.relu(E[tt]+D[uu]);z=head(x);V=z.shape[-1]-5;lp=z[:,:V].log_softmax(-1);dur=z[:,V:].log_softmax(-1);targets=torch.tensor(labels+[0],device=E.device,dtype=torch.long);return lp[:,V-1],lp.gather(1,targets[uu,None])[:,0],dur
def streamed_loss_gradient(head,E,D,labels,scale=1.,backward=True):
 T,U=len(E),len(D);size=T*U;b=np.empty(size);y=np.empty(size);d=np.empty((size,5))
 with torch.no_grad():
  for start in range(0,size,512):
   end=min(size,start+512);v,w,q=selected_lp(head,E,D,labels,start,end);b[start:end]=v.double().cpu().numpy();y[start:end]=w.double().cpu().numpy();d[start:end]=q.double().cpu().numpy()
 loss,gb,gy,gd,gap=tdt_expected(b.reshape(T,U),y.reshape(T,U),d.reshape(T,U,5));assert np.isfinite(loss)and gap<1e-7
 if backward:
  gb=gb.reshape(-1);gy=gy.reshape(-1);gd=gd.reshape(-1,5)
  for start in range(0,size,512):
   end=min(size,start+512);v,w,q=selected_lp(head,E,D,labels,start,end);dev=E.device;dtype=v.dtype;objective=((v*torch.tensor(gb[start:end],device=dev,dtype=dtype)).sum()+(w*torch.tensor(gy[start:end],device=dev,dtype=dtype)).sum()+(q*torch.tensor(gd[start:end],device=dev,dtype=dtype)).sum())*scale;objective.backward()
 return float(loss)
