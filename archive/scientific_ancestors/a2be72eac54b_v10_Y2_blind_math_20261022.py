"""Blind stereo observations; no reference signal, no source-truth claim."""
import numpy as np
import torch

def features(wave,sr=16000,device='cuda'):
    assert wave.ndim==2 and wave.shape[1]==2
    a=torch.as_tensor(wave.T.copy(),dtype=torch.float32,device=device);win=torch.hann_window(512,device=device)
    z=torch.stft(a,512,hop_length=160,win_length=512,window=win,return_complex=True,center=True)
    def smooth(x):
        # Five frames of local evidence, retaining the common physical time.
        return torch.nn.functional.avg_pool1d(x[None],5,stride=1,padding=2,count_include_pad=False)[0]
    l=smooth(abs(z[0])**2);r=smooth(abs(z[1])**2);c=torch.complex(smooth((z[0]*z[1].conj()).real),smooth((z[0]*z[1].conj()).imag));eps=1e-12
    f=torch.fft.rfftfreq(512,1/sr,device=device);low=(f>=150)&(f<1500);high=(f>=1500)&(f<=7500);power=l+r
    coherence=abs(c)**2/(l*r+eps);ild=10*torch.log10((l+eps)/(r+eps));phase=torch.angle(c)
    def avg(x,mask):return (x[mask]*power[mask]).sum(0)/(power[mask].sum(0)+eps)
    cohl=avg(coherence,low);cohh=avg(coherence,high);spatial=avg(torch.sin(phase/2)**2,low)
    # Symmetric analytic equalization-cancellation energy for both orientations.
    # This describes cancellation potential, not target SNR or intelligibility.
    best=torch.ones(z.shape[-1],device=device)*1e20
    for tau in torch.linspace(-.0007,.0007,15,device=device):
        cross=(c*torch.exp(2j*torch.pi*f[:,None]*tau)).real
        gl=torch.clamp(cross/(r+eps),.316,3.162);gr=torch.clamp(cross/(l+eps),.316,3.162)
        # Gain/delay uncertainty prevents spurious infinite cancellation.
        el=l+gl**2*r-2*gl*cross+.04*(l+gl**2*r)
        er=r+gr**2*l-2*gr*cross+.04*(r+gr**2*l)
        ratio=torch.minimum(el/(l+eps),er/(r+eps));best=torch.minimum(best,avg(ratio,low))
    ec=torch.clamp(-10*torch.log10(torch.clamp(best,min=.001)),0,20)/20
    gate=torch.clamp(spatial/.05,0,1);ec=ec*gate
    ildspread=torch.sqrt(torch.clamp(avg(ild**2,low)-avg(ild,low)**2,min=0))/20
    energy=power.sum(0);valid=energy>energy.max()*1e-7
    x=torch.stack([cohl,cohh,spatial,ec,ildspread,avg(abs(ild),low)/20],1);x[~valid]=0
    return x.detach().cpu().numpy(),np.arange(len(x))*.01

def golden(device='cuda'):
    rng=np.random.default_rng(22);x=rng.normal(0,.1,32000).astype(np.float32);mono=np.column_stack([x,x]);same,_=features(mono,device=device);assert same[:,3].max()<1e-5
    stereo=np.column_stack([x,np.roll(x,6)*.7+rng.normal(0,.03,len(x))]).astype(np.float32);a,_=features(stereo,device=device);b,_=features(stereo[:,::-1],device=device);err=float(abs(a-b).max());assert err<1e-4,err
    return dict(same_ear_EC_zero=True,left_right_exchange_max_difference=err,reference_used=False,cancellation_potential_not_target_SNR=True)
