
import numpy as np,hashlib,math
def smooth_gate(length,sr=16000):
 assert length>=1
 edge=min(round(.02*sr),length//2)
 g=np.ones(length,dtype=np.float64)
 if edge:
  ramp=.5-.5*np.cos(np.pi*(np.arange(edge)+1)/(edge+1))
  g[:edge]=ramp;g[-edge:]=ramp[::-1]
 assert g.min()>0 and g.max()<=1
 return g
def local_energy_intervention(x,lo,hi,noise,strength=1.):
 x=np.asarray(x,dtype=np.float32);assert 0<=lo<hi<=len(x)and len(noise)==hi-lo
 if strength==0:return x.copy(),0.
 assert strength==1.
 xx=x[lo:hi].astype(np.float64);g=smooth_gate(len(xx));base=(1-g)*xx;v=g*np.asarray(noise,np.float64)
 A=float(v@v);B=float(base@v);C=float(base@base);E=float(xx@xx);assert A>0 and E-C>=-1e-10*max(E,1)
 a=(math.sqrt(max(B*B+A*max(E-C,0),0))-B)/A
 y=x.copy();y[lo:hi]=(base+a*v).astype(np.float32)
 assert np.isfinite(y).all()
 assert np.array_equal(y[:lo],x[:lo])and np.array_equal(y[hi:],x[hi:])
 return y,a
