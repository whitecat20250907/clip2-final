"""Own decoder q/v low-rank adapter; pretrained weights stay frozen."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=16):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5)); self.scale=alpha/rank

    def forward(self,x):
        original=self.base(x)
        update=(x.float()@self.a.T)@self.b.T
        return original+(update*self.scale).to(original.dtype)


def attach(model,rank=16):
    model.requires_grad_(False); names=[]
    for name,module in list(model.named_modules()):
        if '.decoder.layers.' in name and name.rsplit('.',1)[-1] in ('q_proj','v_proj') and isinstance(module,nn.Linear):
            parent,child=name.rsplit('.',1)
            setattr(model.get_submodule(parent),child,LoRALinear(module,rank,rank)); names.append(name)
    if not names: raise ValueError('No decoder q/v projections found')
    allowed={n for n,p in model.named_parameters() if p.requires_grad}
    if len(allowed)!=2*len(names) or not all(n.endswith(('.a','.b')) for n in allowed):
        raise ValueError('Unexpected trainable base parameter')
    return names
