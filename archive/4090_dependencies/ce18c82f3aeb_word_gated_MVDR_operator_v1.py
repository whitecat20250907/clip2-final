import numpy as np
from mvdr_operator_v1 import stft,inverse,souden_weights,resynthesize
PSEUDOCOUNT=2.
def resynthesize_word_gated(wave,mix,vocal,gates,wrong=False):
 wave=np.asarray(wave,np.float64);mix=np.asarray(mix,np.float64);vocal=np.asarray(vocal,np.float64);N=len(wave);X=np.stack([stft(wave[:,e])for e in [0,1]],-1);V=stft(vocal);B=stft(mix-vocal);pv=abs(V)**2;pn=abs(B)**2;mask=np.divide(pv,pv+pn,out=np.full_like(pv,.5),where=(pv+pn)>1e-30);cov=X[:,:,:,None]*X[:,:,None,:].conj();Rs=(cov*mask[:,:,None,None]).sum(1)/np.maximum(mask.sum(1),1e-30)[:,None,None];Rn=(cov*(1-mask)[:,:,None,None]).sum(1)/np.maximum((1-mask).sum(1),1e-30)[:,None,None];wg,global_meta=souden_weights(Rs,Rn);Zg=np.einsum('fir,fti->ftr',wg.conj(),X);gates=np.asarray(gates,np.float64);assert gates.ndim==2 and gates.shape[1]==X.shape[1]and np.isfinite(gates).all()and gates.min()>=0
 total=gates.sum(0);pi=np.divide(gates,total[None,:],out=np.zeros_like(gates),where=total[None,:]>0);fitg=np.roll(gates,gates.shape[1]//2,axis=1)if wrong else gates;Z=Zg.copy();details=[]
 for j,g in enumerate(fitg):
  a=mask*g[None,:];b=(1-mask)*g[None,:];na=a.sum(1);nb=b.sum(1);rS=((cov*a[:,:,None,None]).sum(1)+PSEUDOCOUNT*Rs)/(na+PSEUDOCOUNT)[:,None,None];rN=((cov*b[:,:,None,None]).sum(1)+PSEUDOCOUNT*Rn)/(nb+PSEUDOCOUNT)[:,None,None];w,meta=souden_weights(rS,rN);Zw=np.einsum('fir,fti->ftr',w.conj(),X);Z+=(Zw-Zg)*pi[j][None,:,None];details.append(dict(word=j,gate_frames=float(g.sum()),target_weight_mean=float(na.mean()),noise_weight_mean=float(nb.mean()),**meta))
 ys=[]
 for e in [0,1]:
  y=inverse(Z[:,:,e],N);rms=np.sqrt(np.mean(y*y));target=np.sqrt(np.mean(wave[:,e]**2))
  if rms>0:y*=target/rms
  else:y=wave[:,e].copy()
  ys.append(y)
 out=np.stack(ys,1);assert out.shape==wave.shape and np.isfinite(out).all()
 return out,dict(wrong=wrong,global_meta=global_meta,words=details,frames=X.shape[1],covered_fraction=float((total>0).mean()),pseudocount=PSEUDOCOUNT)
def qualify_gated():
 rng=np.random.default_rng(257507);v=rng.normal(0,.01,64000);u=rng.normal(0,.02,64000);right=np.r_[np.full(32000,.3),np.full(32000,1.8)];x=np.stack([v+u,v+u*right],1);mix=x.mean(1);T=stft(mix).shape[1];g=np.zeros((2,T));g[0,:T//2]=1;g[1,T//2:]=1
 fixed,_=resynthesize(x,mix,v);const,_=resynthesize_word_gated(x,mix,v,np.ones((3,T)));gap=float(abs(const-fixed).max());assert gap<1e-10
 out,meta=resynthesize_word_gated(x,mix,v,g);bad,_=resynthesize_word_gated(x,mix,v,g,True);swap,_=resynthesize_word_gated(x[:,::-1],mix,v,g);egap=float(abs(swap[:,::-1]-out).max());assert egap<1e-10
 scaled,_=resynthesize_word_gated(3*x,3*mix,3*v,g);sgap=float(abs(scaled-3*out).max());assert sgap<1e-10
 pair=np.stack([v,v],1);ident,_=resynthesize_word_gated(pair,v,.7*v,g);igap=float(abs(ident-pair).max());assert igap<1e-10
 diff=float(np.linalg.norm(out-bad)/np.linalg.norm(out));assert diff>1e-6
 def si(y):
  a=y.mean(1);scale=float(a@v/max(float(a@a),1e-30));return float(np.mean((scale*a-v)**2)/np.mean(v*v))
 return dict(constant_gates_global_replay_gap=gap,ear_swap_gap=egap,shared_scale_gap=sgap,identical_ears_gap=igap,true_wrong_wave_difference=diff,synthetic_fixed_SI_error=si(fixed),synthetic_word_gated_SI_error=si(out),synthetic_wrong_gate_SI_error=si(bad),not_performance_gate_on_synthetic_error=True,meta=meta)
