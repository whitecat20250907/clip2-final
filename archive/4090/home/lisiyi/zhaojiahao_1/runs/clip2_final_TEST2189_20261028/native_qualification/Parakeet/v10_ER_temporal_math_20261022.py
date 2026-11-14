"""Full reference-conditional CTC/TDT time distributions, not greedy timestamps."""
import numpy as np
from scipy.special import logsumexp

def ctc_profile(logp,tokens,blank=0):
    T=len(logp);tokens=np.asarray(tokens,dtype=int);U=len(tokens)
    assert U>0 and np.all(tokens!=blank)
    labels=np.full(2*U+1,blank,dtype=int);labels[1::2]=tokens;S=len(labels)
    emit=logp[:,labels];a=np.full((T,S),-np.inf);a[0,0]=emit[0,0];a[0,1]=emit[0,1]
    for t in range(1,T):
        prev=a[t-1];v=np.stack([prev,np.r_[-np.inf,prev[:-1]],np.r_[[-np.inf]*2,prev[:-2]]])
        skip=(labels!=blank)&(labels!=np.roll(labels,2));skip[:2]=False;v[2,~skip]=-np.inf
        a[t]=emit[t]+logsumexp(v,axis=0)
    z=float(logsumexp(a[-1,-2:]));assert np.isfinite(z),'No finite CTC reference path'
    back=np.full_like(a,-np.inf);back[-1,-2:]=0
    for t in range(T-2,-1,-1):
        v=np.stack([emit[t+1]+back[t+1],np.r_[emit[t+1,1:]+back[t+1,1:],-np.inf],np.r_[emit[t+1,2:]+back[t+1,2:],[-np.inf]*2]])
        skip=np.zeros(S,dtype=bool);skip[:-2]=(labels[2:]!=blank)&(labels[2:]!=labels[:-2]);v[2,~skip]=-np.inf
        back[t]=logsumexp(v,axis=0)
    occupancy=np.exp(a+back-z);assert np.allclose(occupancy.sum(1),1,atol=2e-5)
    token_occ=occupancy[:,1::2].T;mass=token_occ.sum(1);assert np.all(mass>0)
    return token_occ/mass[:,None],dict(logZ=z,token_occupancy=mass.tolist(),state_mass_max_error=float(np.max(np.abs(occupancy.sum(1)-1))))

def tdt_profile(blank_lp,next_lp,duration_lp,durations):
    """Native decoder-start blank; blank duration0 maps to1, last jump saturatesT.

    Conditions on exactly the reference sequence. Other token transitions are
    excluded, not renormalized. Labels can have duration0. Token posterior sums1.
    """
    T,Up=blank_lp.shape;U=Up-1;assert next_lp.shape==(T,U)
    assert duration_lp.shape==(T,U+1,len(durations))
    a=np.full((T+1,U+1),-np.inf);a[0,0]=0
    def edges(t,u):
        for k,d in enumerate(durations):
            yield min(T,t+max(1,d)),u,blank_lp[t,u]+duration_lp[t,u,k],False
            if u<U:yield min(T,t+d),u+1,next_lp[t,u]+duration_lp[t,u,k],True
    for t in range(T):
        for u in range(U+1):
            if not np.isfinite(a[t,u]):continue
            for nt,nu,lp,_ in edges(t,u):a[nt,nu]=np.logaddexp(a[nt,nu],a[t,u]+lp)
    z=float(a[T,U]);assert np.isfinite(z),'No finite TDT reference path'
    b=np.full_like(a,-np.inf);b[T,U]=0;gamma=np.zeros((U,T))
    for t in range(T-1,-1,-1):
        for u in range(U,-1,-1):
            for nt,nu,lp,label in edges(t,u):
                b[t,u]=np.logaddexp(b[t,u],lp+b[nt,nu])
                if label:gamma[u,t]+=np.exp(a[t,u]+lp+b[nt,nu]-z)
    err=float(np.max(np.abs(gamma.sum(1)-1)));assert err<2e-5,(err,z,b[0,0]);assert abs(b[0,0]-z)<2e-5
    return gamma/gamma.sum(1,keepdims=True),dict(logZ=z,token_mass_max_error=err,blank_zero_duration='forced to1 as native generation',terminal='saturate frame jump atT; no labels emitted atT',decoder_start='blank',other_tokens='excluded without renormalizing')

def math_qualification():
    import itertools
    rng=np.random.default_rng(142);logp=rng.normal(size=(4,3));logp-=logsumexp(logp,axis=-1,keepdims=True)
    prof,info=ctc_profile(logp,[1,2]);paths=[];occ=np.zeros((2,4))
    for seq in itertools.product(range(3),repeat=4):
        collapsed=[];owners=[];u=-1;prev=None
        for x in seq:
            if x and x!=prev:collapsed.append(x);u+=1
            owners.append(u if x else -1);prev=x
        if collapsed==[1,2]:
            p=np.exp(sum(logp[t,x] for t,x in enumerate(seq)));paths.append(p)
            for t,u in enumerate(owners):
                if u>=0:occ[u,t]+=p
    assert abs(np.log(sum(paths))-info['logZ'])<1e-10;occ/=occ.sum(1,keepdims=True);assert np.allclose(prof,occ,atol=1e-10)
    T=3;U=2;durs=[0,1,2];tok=rng.normal(size=(T,U+1,4));tok-=logsumexp(tok,axis=-1,keepdims=True);dur=rng.normal(size=(T,U+1,3));dur-=logsumexp(dur,axis=-1,keepdims=True);bl=tok[:,:,3];nxt=np.stack([tok[:,0,1],tok[:,1,2]],1)
    pred,inf=tdt_profile(bl,nxt,dur,durs);total=0.;counts=np.zeros((U,T));number=0
    def visit(t,u,p,emissions):
        nonlocal total,number
        if t==T:
            if u==U:
                total+=p;number+=1
                for j,s in enumerate(emissions):counts[j,s]+=p
            return
        for k,d in enumerate(durs):
            visit(min(T,t+max(1,d)),u,p*np.exp(bl[t,u]+dur[t,u,k]),emissions)
            if u<U:visit(min(T,t+d),u+1,p*np.exp(nxt[t,u]+dur[t,u,k]),emissions+[t])
    visit(0,0,1.,[]);assert abs(np.log(total)-inf['logZ'])<1e-10;assert np.allclose(counts/total,pred,atol=1e-10)
    return dict(CTC='brute-force path sum and full occupancy agree <1e-10',TDT='brute-force path sum and every token full-time marginal agree <1e-10',TDT_paths=number)

if __name__=='__main__':print(math_qualification())
