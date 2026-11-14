"""Prespecified time-distribution relations plus unchanged MF/MOSS relations."""
import numpy as np
STREAMS=['A1L','A1R','A2L','A2R']
def effective_hop(meta,model):
    hop=float(meta['frame_hop_seconds'])
    if model=='Granite':
        assert abs(hop-.04)<1e-7 or abs(hop-.08)<1e-7
        # Old extraction metadata omitted native frame_stacking=2. Raw arrays
        # are correct and stay untouched; correct the cached time coordinate.
        return .08
    assert abs(hop-.08)<1e-7
    return hop
def grid(q,hop,duration,bins=128):
    ix=np.minimum(bins-1,np.maximum(0,(((np.arange(q.shape[1])+.5)*hop/duration)*bins).astype(int)));out=np.zeros((len(q),bins))
    for k in range(len(q)):out[k]=np.bincount(ix,weights=q[k],minlength=bins)
    return out
def entropy(q):return -(q*np.log(q+1e-30)).sum(-1)/np.log(q.shape[-1])
def compare(a,b):
    m=.5*(a+b);js=.5*((a*np.log((a+1e-30)/(m+1e-30))).sum(-1)+(b*np.log((b+1e-30)/(m+1e-30))).sum(-1))/np.log(2);x=(np.arange(a.shape[-1])+.5)/a.shape[-1]
    return np.sqrt(a*b).sum(-1),js,np.abs((a*x).sum(-1)-(b*x).sum(-1)),np.abs(entropy(a)-entropy(b)),np.abs(a.argmax(-1)-b.argmax(-1))/a.shape[-1]
def adjacent(a):return np.r_[np.sqrt(a[:-1]*a[1:]).sum(-1),0.]
def row_features(g,p,gj,pj,shuffle=False):
    duration=max(v['duration_seconds'] for v in gj['streams'].values());n=len(g['A1L_valid']);temporal=[];internal=[];pairs={};grids={};internal_changes={}
    for stream in STREAMS:
        gm=gj['streams'][stream];pm=pj['streams'][stream];assert gm['PCM_SHA']==pm['PCM_SHA'];gh=effective_hop(gm,'Granite');ph=effective_hop(pm,'Parakeet');assert abs(g[stream+'_final'].shape[1]*gh-gm['duration_seconds'])<2*gh and abs(p[stream+'_final'].shape[1]*ph-pm['duration_seconds'])<2*ph;gg=grid(g[stream+'_final'],gh,duration);pg=grid(p[stream+'_final'],ph,duration);mid=grid(g[stream+'_mid'],gh,duration);gv=g[stream+'_valid'];pv=p[stream+'_valid'];ml=np.asarray([0 if z is None else z for z in gm['mid']['word_native_logp_mean']]);fl=np.asarray([0 if z is None else z for z in gm['final']['word_native_logp_mean']]);mprob=np.exp(np.clip(ml,-20,0));fprob=np.exp(np.clip(fl,-20,0))
        if shuffle:pg=np.roll(pg,1,axis=0);pv=np.roll(pv,1);mid=np.roll(mid,1,axis=0);mprob=np.roll(mprob,1)
        valid=gv&pv;metrics=compare(gg,pg);temporal.extend([*metrics,np.abs(adjacent(gg)-adjacent(pg)),valid.astype(float)]);pairs[stream]=metrics;grids[stream]=(gg,pg)
        im=compare(mid,gg);internal.extend([im[0],im[1],im[2],entropy(gg)-entropy(mid),fprob-mprob,mprob*(1-fprob),(1-mprob)*fprob]);internal_changes[stream]=im
    for ear in ['L','R']:temporal.extend([pairs['A1'+ear][0]-pairs['A2'+ear][0],pairs['A1'+ear][1]-pairs['A2'+ear][1]])
    for ear in ['L','R']:temporal.append(compare(grids['A1'+ear][0],grids['A2'+ear][0])[0]-compare(grids['A1'+ear][1],grids['A2'+ear][1])[0])
    for source in ['A1','A2']:temporal.append(pairs[source+'L'][0]-pairs[source+'R'][0])
    tx=np.stack(temporal,-1).astype('float32');ix=np.stack(internal,-1).astype('float32');assert tx.shape==(n,36) and ix.shape==(n,28) and np.isfinite(tx).all() and np.isfinite(ix).all();return tx,ix
