import numpy as np
from scipy import signal
FS=16000;NFFT=512;HOP=128
def stft(x):
 return signal.stft(x,fs=FS,window='hann',nperseg=NFFT,noverlap=NFFT-HOP,nfft=NFFT,boundary='zeros',padded=True,axis=0)[2]
def inverse(z,N):
 y=signal.istft(z,fs=FS,window='hann',nperseg=NFFT,noverlap=NFFT-HOP,nfft=NFFT,boundary=True,time_axis=-1,freq_axis=0)[1]
 assert len(y)>=N
 return y[:N]
def souden_weights(Rs,Rn):
 assert Rs.shape==Rn.shape and Rs.shape[-2:]==(2,2)
 scale=np.maximum(np.trace(Rn,axis1=-2,axis2=-1).real/2,1e-30)
 loaded=Rn+(.001*scale)[...,None,None]*np.eye(2)
 T=np.linalg.solve(loaded,Rs);tr=np.trace(T,axis1=-2,axis2=-1)
 assert np.isfinite(T).all()
 good=(tr.real>1e-12)&(abs(tr.imag)<=1e-6*np.maximum(abs(tr.real),1e-30))
 w=np.broadcast_to(np.eye(2,dtype=np.complex128),T.shape).copy()
 w[good]=T[good]/tr[good,None,None].real
 return w,dict(fallback_bins=int((~good).sum()),total_bins=len(good))
def resynthesize(wave,mix,vocal,shift=False):
 wave=np.asarray(wave,dtype=np.float64);mix=np.asarray(mix,dtype=np.float64);vocal=np.asarray(vocal,dtype=np.float64);N=len(wave)
 assert wave.shape==(N,2) and mix.shape==vocal.shape==(N,)
 X=np.stack([stft(wave[:,e])for e in [0,1]],-1)
 V=stft(vocal);B=stft(mix-vocal);pv=abs(V)**2;pn=abs(B)**2;den=pv+pn
 mask=np.divide(pv,den,out=np.full_like(pv,.5),where=den>1e-30)
 if shift:mask=np.roll(mask,mask.shape[1]//2,axis=1)
 cov=X[:,:,:,None]*X[:,:,None,:].conj()
 Rs=(cov*mask[:,:,None,None]).sum(1)/np.maximum(mask.sum(1),1e-30)[:,None,None]
 Rn=(cov*(1-mask)[:,:,None,None]).sum(1)/np.maximum((1-mask).sum(1),1e-30)[:,None,None]
 w,meta=souden_weights(Rs,Rn);Z=np.einsum('fir,fti->ftr',w.conj(),X);ys=[]
 for e in [0,1]:
  y=inverse(Z[:,:,e],N);rms=np.sqrt(np.mean(y*y));target=np.sqrt(np.mean(wave[:,e]**2))
  if rms>0:y*=target/rms
  else:y=wave[:,e].copy()
  assert np.isfinite(y).all();ys.append(y)
 return np.stack(ys,1),dict(**meta,mask_mean=float(mask.mean()),mask_std=float(mask.std()),shift=shift)
def synthetic_qualification():
 rng=np.random.default_rng(20261022);x=rng.normal(0,.01,32001);v=.7*x;pair=np.stack([x,x],1)
 y,meta=resynthesize(pair,x,v);identity=float(abs(y-pair).max());assert identity<1e-10
 recon=inverse(stft(x),len(x));gap=float(abs(x-recon).max());assert gap<1e-12
 steering=np.array([1,.7*np.exp(.5j)]);Rs=np.outer(steering,steering.conj())[None];Rn=np.array([[[1.,.1],[.1,2.]]],dtype=np.complex128)
 w,_=souden_weights(Rs,Rn);response=np.einsum('ir,i->r',w[0].conj(),steering);assert np.max(abs(response-steering))<1e-10
 noise=np.einsum('ir,ij,jr->r',w[0].conj(),Rn[0],w[0]).real;assert noise[0]<Rn[0,0,0].real
 return dict(identity_maxgap=identity,STFT_ISTFT_gap=gap,distortionless_response_maxgap=float(abs(response-steering).max()),reference_noise=float(Rn[0,0,0].real),MVDR_noise=float(noise[0]))
