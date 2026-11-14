from mvdr_native_core_v1 import *
from word_gated_MVDR_operator_v1 import resynthesize_word_gated,qualify_gated,stft
def gates(s,N,T):
 nw=len(assets[s]['words']);j=idx[s];a=al['weights'][al['offsets'][j]:al['offsets'][j+1],:nw].astype(np.float64);G=len(a);centers=(np.arange(G)+.5)*N/G;time=np.arange(T)*128;out=np.zeros((nw,T));details=[]
 for w in range(nw):
  mass=np.maximum(a[:,w],0)
  if mass.sum()<=0:out[w]=1.;details.append(dict(word=w,fallback_global=True));continue
  c=np.cumsum(mass)/mass.sum();lo=max(0,int(np.searchsorted(c,.05))-1);hi=min(G-1,int(np.searchsorted(c,.95))+1);v=np.interp(time,centers,mass,left=0.,right=0.);v[(time<lo*N/G)|(time>=(hi+1)*N/G)]=0
  if v.max()<=0:v[(time>=lo*N/G)&(time<(hi+1)*N/G)]=1
  assert v.max()>0;out[w]=v/v.max();details.append(dict(word=w,lo=lo,hi=hi,Granite_frames=G,gate_weight=float(out[w].sum())))
 return out,details
def measure_gated(s,global12,proof=False):
 x=stereo(s);mix,vocal=component(s);assert len(x)==len(mix)==len(vocal);T=stft(mix).shape[1];g,gd=gates(s,len(x),T);good,meta=resynthesize_word_gated(x,mix,vocal,g);bad,wmeta=resynthesize_word_gated(x,mix,vocal,g,True);ps=[]
 for wave in [good[:,0],good[:,1],bad[:,0],bad[:,1]]:ps.append(profiles(s,probability_audio(np.ascontiguousarray(wave,dtype=np.float32))))
 a=sym(ps[0],ps[1]);b=sym(ps[2],ps[3]);vals=dict(global24=np.c_[global12,np.zeros((12,12))],local24=np.c_[global12,a-global12],wrong24=np.c_[global12,b-global12])
 for v in vals.values():v[len(assets[s]['words']):]=0;assert v.shape==(12,24)and np.isfinite(v).all()
 rec=dict(signal=s,gates=gd,local_meta=meta,wrong_meta=wmeta)
 if proof:
  original,_=resynthesize(x,mix,vocal);pp=[profiles(s,probability_audio(np.ascontiguousarray(original[:,e],dtype=np.float32)))for e in [0,1]];gap=float(abs(sym(*pp)-global12).max());assert gap<=.002;rec['global_native_cache_replay_gap']=gap
 return vals,rec
