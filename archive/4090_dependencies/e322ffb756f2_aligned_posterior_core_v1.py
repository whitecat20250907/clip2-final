import numpy as np,torch,math
from scipy.special import logsumexp
from teacher_support_core_v1 import encoded
from tdt_loss_core_v1 import tdt_expected
def _pool(blank,target,duration,stats,owner,nwords):
 T,U=blank.shape;L=U-1;loss,gb,gy,gd,gap=tdt_expected(blank,target,duration);A=-gy[:,:L]*(T+L);mass=float(np.max(abs(A.sum(0)-1)));assert mass<1e-6 and gap<1e-7;pos=np.arange(T)/max(T-1,1);mean=(A*pos[:,None]).sum(0);std=np.sqrt(np.maximum((A*(pos[:,None]-mean[None,:])**2).sum(0),0));entropy=-(A*np.log(np.maximum(A,1e-300))).sum(0)/math.log(max(T,2));token=np.c_[(A[:,:,None]*stats[:,:L]).sum(0),entropy,mean,std,np.full(L,-loss)];assert token.shape==(L,16);out=np.zeros((nwords,32))
 for j in range(nwords):
  ii=np.where(owner==j)[0]
  if len(ii):out[j,:16]=token[ii].mean(0);out[j,16:]=token[ii].min(0)
 assert np.isfinite(out).all();return out,mass
def extract32(engine,head,array,words,dense=False):
 device=engine.device;E=engine.net['encoder_projector'](torch.tensor(array,dtype=torch.float32,device=device));zero=engine.net['encoder_projector'](torch.zeros(1,1024,device=device));outputs=[];masses=[];V=8193
 for case in [0,1]:
  labels,owner=encoded(engine,words,case);L=len(labels);U=L+1;T=len(array);target_ids=torch.tensor(labels+[0],dtype=torch.long,device=device);D=engine.net['decoder'](torch.tensor([[8192]+labels],dtype=torch.long,device=device),cache=None)[0];null_z=head(torch.relu(zero+D));null_lp=null_z[:,:V].log_softmax(-1)
  if dense:
   z=head(torch.relu(E[:,None,:]+D[None,:,:])).cpu().numpy().astype(np.float64);nz=null_z.cpu().numpy().astype(np.float64);lp=z[:,:,:V]-logsumexp(z[:,:,:V],axis=-1,keepdims=True);nlp=nz[:,:V]-logsumexp(nz[:,:V],axis=-1,keepdims=True);du=z[:,:,V:]-logsumexp(z[:,:,V:],axis=-1,keepdims=True);P=np.exp(lp);Q=np.exp(nlp)[None,:,:];mix=np.logaddexp(lp,nlp[None,:,:])-math.log(2);yy=np.take_along_axis(lp,np.asarray(labels+[0])[None,:,None].repeat(T,axis=0),-1)[...,0];bb=lp[:,:,8192];ny=nlp[np.arange(U),labels+[0]][None,:].repeat(T,axis=0);top=np.partition(lp,-2,axis=-1);margin=top[:,:,-1]-top[:,:,-2]
   stats=np.stack([yy,yy-ny,ny,yy-bb,-(P*lp).sum(-1)/math.log(V),margin,np.maximum((P*(lp-nlp)).sum(-1),0),np.maximum((Q*(nlp-lp)).sum(-1),0),np.maximum(.5*((P*(lp-mix)).sum(-1)+(Q*(nlp-mix)).sum(-1)),0),.5*abs(P-Q).sum(-1),(np.exp(du)*np.arange(5)).sum(-1)/4,-(np.exp(du)*du).sum(-1)/math.log(5)],axis=-1);value,mass=_pool(bb,yy,du,stats,owner,len(words))
  else:
   size=T*U;b=np.empty(size);y=np.empty(size);dur=np.empty((size,5));stats=np.empty((size,12));null_y=null_lp.gather(1,target_ids[:,None])[:,0]
   for start in range(0,size,512):
    end=min(size,start+512);index=torch.arange(start,end,device=device);ti=torch.div(index,U,rounding_mode='floor');ui=index%U;z=head(torch.relu(E[ti]+D[ui]));lp=z[:,:V].log_softmax(-1);du=z[:,V:].log_softmax(-1);yy=lp.gather(1,target_ids[ui,None])[:,0];bb=lp[:,8192];ny=null_y[ui];rr=lp.double();nn=null_lp[ui].double();P=rr.exp();Q=nn.exp();mix=torch.logaddexp(rr,nn)-math.log(2);top=rr.topk(2,dim=-1).values;dp=du.double();vals=torch.stack([yy.double(),(yy-ny).double(),ny.double(),(yy-bb).double(),-(P*rr).sum(-1)/math.log(V),top[:,0]-top[:,1],(P*(rr-nn)).sum(-1).clamp_min(0),(Q*(nn-rr)).sum(-1).clamp_min(0),(.5*((P*(rr-mix)).sum(-1)+(Q*(nn-mix)).sum(-1))).clamp_min(0),.5*(P-Q).abs().sum(-1),(dp.exp()*torch.arange(5,device=device)).sum(-1)/4,-(dp.exp()*dp).sum(-1)/math.log(5)],-1);b[start:end]=bb.double().cpu().numpy();y[start:end]=yy.double().cpu().numpy();dur[start:end]=dp.cpu().numpy();stats[start:end]=vals.cpu().numpy()
   value,mass=_pool(b.reshape(T,U),y.reshape(T,U),dur.reshape(T,U,5),stats.reshape(T,U,12),owner,len(words))
  outputs.append(value);masses.append(mass)
 return .5*(outputs[0]+outputs[1]),max(masses)
