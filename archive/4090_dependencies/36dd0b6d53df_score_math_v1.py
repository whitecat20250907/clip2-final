import numpy as np
from scipy.special import logsumexp
def greedy(lp):
 ids=np.argmax(lp,axis=1).tolist();out=[]
 for i,v in enumerate(ids):
  if v and (i==0 or v!=ids[i-1]):out.append(int(v))
 return out
def edits(ref,hyp):
 nr,nh=len(ref),len(hyp);d=np.empty((nr+1,nh+1),np.int64);d[:,0]=np.arange(nr+1);d[0]=np.arange(nh+1)
 for i in range(1,nr+1):
  for j in range(1,nh+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
 i,j=nr,nh;match=sub=dele=ins=0
 while i or j:
  if i and j and ref[i-1]==hyp[j-1] and d[i,j]==d[i-1,j-1]:match+=1;i-=1;j-=1
  elif i and j and d[i,j]==d[i-1,j-1]+1:sub+=1;i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:dele+=1;i-=1
  else:assert j and d[i,j]==d[i,j-1]+1;ins+=1;j-=1
 assert match+sub+dele==nr and match+sub+ins==nh and sub+dele+ins==d[-1,-1]
 return match,sub,dele,ins
def minimum(seq):return len(seq)+sum(a==b for a,b in zip(seq,seq[1:]))
def ridge_fit(x,y,lam=10.):
 med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
 fill=np.where(np.isfinite(x),x,med);mu=fill.mean(0);sd=fill.std(0).clip(1e-4)
 z=np.clip((fill-mu)/sd,-6,6);a=np.column_stack([np.ones(len(z)),z]);pen=np.diag(np.r_[0.,np.full(x.shape[1],lam)])
 coef=np.linalg.solve(a.T@a+pen,a.T@y)
 return dict(median=med,mean=mu,std=sd,coef=coef)
def ridge_predict(state,x):
 fill=np.where(np.isfinite(x),x,state['median']);z=np.clip((fill-state['mean'])/state['std'],-6,6)
 return np.clip(np.column_stack([np.ones(len(z)),z])@state['coef'],0,1)
