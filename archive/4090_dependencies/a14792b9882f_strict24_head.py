"""Equal architecture for public base/adapted LP controls, one word readout."""
import torch
from torch import nn

class TFReadout(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.severity=nn.Embedding(3,8)
        self.context=nn.Sequential(nn.Linear(261+1+8+2,width),nn.GELU())
        self.listener=nn.Sequential(nn.Linear(3,width),nn.GELU())
        self.output=nn.Linear(width,1);self.listener_output=nn.Linear(width,1)
        for layer in (self.output,self.listener_output):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,logits,evidence,listener,support,words,severity,mode):
        assert mode in ('context_only','listener_additive')
        assert support.dtype==words.dtype==torch.bool and not bool((support&~words).any())
        n=words.sum(1);assert bool((n>0).all())
        position=(torch.arange(words.shape[1],device=words.device)[None,:]+.5)/n[:,None]
        inverse=n.float().reciprocal()[:,None].expand_as(position)
        sev=self.severity(severity)[:,None].expand(-1,words.shape[1],-1)
        context=self.context(torch.cat([evidence,logits[...,None]/4,sev,position[...,None],inverse[...,None]],-1))
        z=self.listener(torch.zeros_like(listener) if mode=='context_only' else listener)
        # Local-matcher availability gates local context, not independent decoder evidence.
        # The verified bank provides decoder evidence for every nonpadding word.
        # Same hidden width and parameters; supported word context conditions listener evidence.
        z=z*(1+torch.tanh(context)*support[...,None])
        delta=.25*torch.tanh(self.output(context).squeeze(-1)*support+self.listener_output(z).squeeze(-1))*words
        updated=logits+delta
        return (torch.sigmoid(updated)*words).sum(1)/n,updated


"""Free shared word readout branch; no condition expert, routing or anchoring.

The same architecture receives either existing evidence, canonical CTC,
canonical plus lexical competition, or deliberately misbound competition.
All inherited readout weights remain trainable. New branch logits are not
clipped, support gated, shrunk to a fixed residual bound or parent penalized.
"""
import torch
from torch import nn

class CTCReadout(nn.Module):
    def __init__(self,base,seed):
        super().__init__();self.base=base
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.branch=nn.Sequential(nn.Linear(39,32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,1))
            nn.init.zeros_(self.branch[-1].weight);nn.init.zeros_(self.branch[-1].bias)

    def forward(self,logits,evidence,listener,support,words,severity,extra):
        if extra.shape!=(*words.shape,24) or not bool(torch.isfinite(extra).all()):raise ValueError('Incomplete24D public evidence')
        _,old_logits=self.base(logits,evidence,listener,support,words,severity,'listener_additive')
        n=words.sum(1);position=(torch.arange(words.shape[1],device=words.device)[None,:]+.5)/n[:,None]
        inverse=n.float().reciprocal()[:,None].expand_as(position)
        sev=self.base.severity(severity)[:,None].expand(-1,words.shape[1],-1)
        x=torch.cat((extra,logits[...,None]/4,listener,sev,position[...,None],inverse[...,None],support.float()[...,None]),-1)
        updated=old_logits+self.branch(x).squeeze(-1)
        probability=updated.sigmoid()*words
        return probability.sum(1)/n,updated

def wrong_word_competition(features,support,lengths):
    """Rotate competition only across supported legal words within each row.

    Keep target canonical evidence and public neighbour availability exact.
    No row is discarded; no label/residual selects source or destination.
    """
    out=features.clone();changed=0
    for i,n in enumerate(lengths):
        ix=torch.nonzero(support[i,:int(n)],as_tuple=False).flatten()
        if len(ix)>1:
            out[i,ix,18:]=features[i,torch.roll(ix,1),18:]
            changed+=int((out[i,ix,18:]!=features[i,ix,18:]).any(-1).sum())
    return out,changed


"""Small own word-level correction; real neighbour masks, no peer predictions."""
import torch
from torch import nn


class FineResidual(nn.Module):
    def __init__(self,parent,seed,mode='bounded_local'):
        super().__init__()
        if mode not in ('bounded_local','bounded_pooled','logit_local','bounded_local_joint'):
            raise ValueError(mode)
        self.parent,self.mode=parent,mode
        self.freeze_parent=mode!='bounded_local_joint'
        if self.freeze_parent:
            for p in parent.parameters(): p.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.condition=nn.Sequential(nn.Linear(65,32),nn.GELU(),nn.Linear(32,1))
            nn.init.zeros_(self.condition[-1].weight)
            nn.init.zeros_(self.condition[-1].bias)
        self.last_delta=self.last_parent_probability=None

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_parent: self.parent.eval()
        return self

    def forward(self,logits,evidence,listener,support,words,severity,extra):
        _,z=self.parent(logits,evidence,listener,support,words,severity,extra)
        n=words.sum(1)
        position=(torch.arange(words.shape[1],device=words.device)[None,:]+.5)/n[:,None]
        inverse=n.float().reciprocal()[:,None].expand_as(position)
        sev=self.parent.base.severity(severity)[:,None].expand(-1,words.shape[1],-1)
        x=torch.cat((extra,logits[...,None]/4,listener,sev,position[...,None],
                     inverse[...,None],support.float()[...,None]),-1)
        hidden=self.parent.branch[:-1](x)
        mask=words[...,None]
        if self.mode=='bounded_pooled':
            mean=(hidden*mask).sum(1)/n[:,None]
            var=((hidden-mean[:,None]).square()*mask).sum(1)/n[:,None]
            c1=mean[:,None].expand_as(hidden)
            c2=var.clamp_min(1e-12).sqrt()[:,None].expand_as(hidden)
        else:
            left=torch.zeros_like(hidden);right=torch.zeros_like(hidden)
            lm=torch.zeros_like(mask);rm=torch.zeros_like(mask)
            left[:,1:]=hidden[:,:-1];lm[:,1:]=mask[:,:-1]
            right[:,:-1]=hidden[:,1:];rm[:,:-1]=mask[:,1:]
            count=lm.float()+rm.float()
            c1=(left*lm+right*rm)/count.clamp_min(1)
            c1=torch.where(count>0,c1,hidden)
            c2=(hidden-c1).abs()
        p=z.sigmoid()
        h=torch.cat((hidden,c1,c2,listener,sev,support.float()[...,None],
                     position[...,None],inverse[...,None],z[...,None]/4,
                     p[...,None],(4*p*(1-p))[...,None]),-1)
        assert h.shape[-1]==65
        r=self.condition(h).squeeze(-1)
        if self.mode=='logit_local':
            updated=z+r
            q=updated.sigmoid()
        else:
            q=p+.2*p*(1-p)*r.tanh()
            updated=torch.logit(q.clamp(1e-7,1-1e-7))
        self.last_delta=(q-p).detach()
        self.last_parent_probability=p.detach()
        return (q*words).sum(1)/n,updated


"""Count likelihoods; learned observation dispersion does not change inference mean by itself."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def recover_counts(scores,lengths):
    scores=np.asarray(scores,np.float64);lengths=np.asarray(lengths)
    raw=scores*lengths;counts=np.rint(raw).astype(np.int64)
    if not ((lengths>0).all() and np.isfinite(raw).all() and np.max(abs(raw-counts))<1e-6 and (counts>=0).all() and (counts<=lengths).all()):
        raise ValueError('Official aggregate score does not certify an integer count on the legal scoring grid')
    return counts

def logpmf(counts,lengths,mean,concentration=None):
    k=counts.double();n=lengths.double();mu=mean.double().clamp(1e-7,1.-1e-7)
    choose=torch.lgamma(n+1)-torch.lgamma(k+1)-torch.lgamma(n-k+1)
    if concentration is None:return choose+k*mu.log()+(n-k)*torch.log1p(-mu)
    phi=concentration.double();a=mu*phi;b=(1-mu)*phi
    return choose+torch.lgamma(k+a)+torch.lgamma(n-k+b)-torch.lgamma(n+phi)+torch.lgamma(phi)-torch.lgamma(a)-torch.lgamma(b)

def count_loss(counts,lengths,mean,concentration=None):
    return (-logpmf(counts,lengths,mean,concentration)/lengths).mean()

class OverdispersedReadout(nn.Module):
    def __init__(self,mean_model,concentration):
        super().__init__();self.mean_model=mean_model
        assert .001<concentration<100000.
        value=concentration-.001
        inverse=value if value>20 else float(np.log(np.expm1(value)))
        self.eta=nn.Parameter(torch.tensor(inverse,dtype=torch.float32,device=next(mean_model.parameters()).device))

    def concentration(self):return F.softplus(self.eta)+.001

    def forward(self,*args):return self.mean_model(*args)

def functional_check():
    from scipy.special import gammaln
    gap=normgap=meangap=vargap=0.
    for n0 in [5,7,12]:
        for mu0 in [.01,.2,.6,.99]:
            for phi0 in [.25,1.8,12.]:
                k=torch.arange(n0+1,dtype=torch.float64);n=torch.full_like(k,n0);mu=torch.full_like(k,mu0,requires_grad=True);phi=torch.tensor(phi0,dtype=torch.float64,requires_grad=True)
                lp=logpmf(k,n,mu,phi);probs=lp.exp();reference=gammaln(n0+1)-gammaln(k.numpy()+1)-gammaln(n0-k.numpy()+1)+gammaln(k.numpy()+mu0*phi0)+gammaln(n0-k.numpy()+(1-mu0)*phi0)-gammaln(n0+phi0)+gammaln(phi0)-gammaln(mu0*phi0)-gammaln((1-mu0)*phi0)
                gap=max(gap,float(abs(lp.detach().numpy()-reference).max()));normgap=max(normgap,abs(float(probs.sum().detach())-1.))
                mean=(probs*k/n0).sum();meangap=max(meangap,abs(float(mean.detach())-mu0));variance=(probs*(k/n0-mu0).square()).sum();expected=mu0*(1-mu0)*(n0+phi0)/(n0*(1+phi0));vargap=max(vargap,abs(float(variance.detach())-expected))
                (-lp.mean()).backward();assert torch.isfinite(mu.grad).all() and torch.isfinite(phi.grad)
    assert gap<1e-9 and normgap<1e-9 and meangap<1e-9 and vargap<1e-9
    means=torch.tensor([.2,.7],requires_grad=True);counts=torch.tensor([0,5]);n=torch.tensor([7,7]);small=count_loss(counts,n,means,torch.tensor(1.8));big=count_loss(counts,n,means,None)
    a=torch.autograd.grad(small,means,retain_graph=True)[0];b=torch.autograd.grad(big,means)[0];assert not torch.allclose(a,b)
    try:recover_counts([.51],[7])
    except ValueError:pass
    else:raise AssertionError('Cannot invent off-grid integer labels')
    assert recover_counts([0.,3/7,1.],[7,7,7]).tolist()==[0,3,7]
    return dict(passed=True,independent_logPMF_gap=gap,normalization_gap=normgap,expected_mean_gap=meangap,variance_gap=vargap,mean_gradients_depend_on_dispersion=True,changing_dispersion_only_leaves_expected_score_unchanged=True,off_grid_labels_rejected=True,labels_read=False)
