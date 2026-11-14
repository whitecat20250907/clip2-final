"""Frozen conditional target support; no path likelihood or encoder execution."""
import numpy as np,torch,math
from scipy.special import logsumexp
from torch.nn import functional as F
def encoded(engine,words,case):
 ww=list(words)
 if case:ww[0]=ww[0][:1].upper()+ww[0][1:]
 text=' '.join(ww);enc=engine.tok.encode(text,add_special_tokens=False);ids=enc.ids
 assert ids and all(0<=v<engine.V and v!=engine.blank and v!=engine.eos for v in ids)
 spans=[];off=0
 for w in ww:spans.append((off,off+len(w)));off+=len(w)+1
 owners=[]
 for a,b in enc.offsets:
  overlap=[max(0,min(b,y)-max(a,x))for x,y in spans];m=max(overlap)
  owners.append(int(np.argmax(overlap))if m else next((i for i,(x,y)in enumerate(spans)if x>=b),len(spans)-1))
 return ids,np.asarray(owners)
def extract(engine,array,words,dense=False):
 pieces=[];retained=[]
 E=engine.net['encoder_projector'](torch.tensor(array,dtype=torch.float32,device=engine.device))
 zero=engine.net['encoder_projector'](torch.zeros(1,1024,dtype=torch.float32,device=engine.device))
 for case in [0,1]:
  ids,owners=encoded(engine,words,case);targets=torch.tensor(ids,device=engine.device,dtype=torch.long)
  prefix=torch.tensor([[engine.blank]+ids],device=engine.device,dtype=torch.long);D=engine.net['decoder'](prefix,cache=None)[0,:len(ids)]
  def lp(z,t):
   v=z[...,:engine.V].double();den=torch.logsumexp(v,-1)
   return (v.gather(-1,t[...,None])[...,0]-den).cpu().numpy(),(v[...,engine.blank]-den).cpu().numpy()
  nz=engine.net['joint'](encoder_hidden_states=zero.expand(len(ids),-1),decoder_hidden_states=D);null,_=lp(nz,targets)
  T,L=len(array),len(ids)
  if dense:
   z=F.linear(F.relu(E[:,None,:]+D[None,:,:]),engine.net['joint'].head.weight,engine.net['joint'].head.bias);real,blank=lp(z,targets[None,:].expand(T,-1))
  else:
   real=np.empty((T,L));blank=np.empty((T,L));size=T*L
   for start in range(0,size,512):
    k=torch.arange(start,min(size,start+512),device=engine.device);ti=torch.div(k,L,rounding_mode='floor');wi=k%L
    z=engine.net['joint'](encoder_hidden_states=E[ti],decoder_hidden_states=D[wi]);a,b=lp(z,targets[wi]);real.reshape(-1)[start:start+len(k)]=a;blank.reshape(-1)[start:start+len(k)]=b
  diff=real-null[None,:]
  stats=np.stack([logsumexp(real,axis=0)-math.log(T),real.max(0),diff.max(0),diff.mean(0),(real-blank).max(0),(diff>0).mean(0)],axis=1)
  out=np.zeros((len(words),16))
  for i in range(len(words)):
   jj=np.where(owners==i)[0]
   if len(jj):
    out[i,:6]=stats[jj].mean(0);out[i,6:12]=stats[jj].min(0);out[i,12]=null[jj].mean();out[i,13]=null[jj].min();out[i,14]=math.log1p(len(jj));out[i,15]=float(any(ids[j]==0 for j in jj))
   else:out[i,15]=1.
  assert np.isfinite(out).all();pieces.append(out);retained.append((real,blank,null))
 return np.c_[pieces[0],pieces[1]],retained
