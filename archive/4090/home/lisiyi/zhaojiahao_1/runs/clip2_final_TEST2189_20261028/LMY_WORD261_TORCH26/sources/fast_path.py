"""Same ordered graph on CPU arrays: avoid thousands of tiny GPU launches."""
import numpy as np
import torch


def numpy_posterior(emissions, temperature=.2):
    x = np.asarray(emissions,np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not temperature > 0:
        raise ValueError('Invalid emissions')
    t,n = x.shape
    if not 0 < n <= t: raise ValueError('Invalid path lengths')
    states = 2*n+1
    e = np.zeros((t,states),np.float64); e[:,1::2] = x/temperature
    a = np.full_like(e,-np.inf); a[0,0] = 0.; a[0,1] = e[0,1]
    target_skip = (np.arange(states)%2 == 1) & (np.arange(states)>=3)
    source_skip = (np.arange(states)%2 == 1) & (np.arange(states)<states-2)
    for i in range(1,t):
        prev = a[i-1]
        one = np.r_[-np.inf,prev[:-1]]
        two = np.r_[-np.inf,-np.inf,prev[:-2]]; two[~target_skip] = -np.inf
        a[i] = np.logaddexp(np.logaddexp(prev,one),two)+e[i]
    z = np.logaddexp(a[-1,-2],a[-1,-1])
    b = np.full_like(e,-np.inf); b[-1,-2:] = 0.
    for i in range(t-2,-1,-1):
        nxt = b[i+1]+e[i+1]
        one = np.r_[nxt[1:],-np.inf]
        two = np.r_[nxt[2:],-np.inf,-np.inf]; two[~source_skip] = -np.inf
        b[i] = np.logaddexp(np.logaddexp(nxt,one),two)
    all_states = np.exp(a+b-z); p = all_states[:,1::2]
    gap = float(abs(all_states.sum(1)-1).max()); minimum = float(p.sum(0).min())
    if not np.isfinite(p).all() or gap > 1e-7 or minimum < 1-1e-7:
        raise ValueError('Invalid fast posterior')
    return p.astype(np.float32),float(z),dict(state_mass_max_gap=gap,minimum_phone_frame_mass=minimum)


@torch.no_grad()
def fast_posterior(emissions,temperature=.2):
    p,z,diag = numpy_posterior(emissions.detach().cpu().numpy(),temperature)
    return torch.from_numpy(p).to(emissions.device),torch.tensor(z,dtype=torch.float64,device=emissions.device),diag
