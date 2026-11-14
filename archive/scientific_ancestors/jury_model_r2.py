"""Manifest-driven, word-level witness set attention; no temporal pooling."""
import torch
from torch import nn

WITNESSES = (
    ('legacy_asr_rules', 0, 65, 0, 0),
    ('n2_medium', 65, 73, 22, 34),
    ('ce_medium', 73, 81, 34, 46),
    ('l7_large_v3', 81, 89, 46, 58),
    ('mixture_music', 89, 103, 0, 0),
    ('isolated_vocals', 103, 119, 0, 0),
)

class JuryModel(nn.Module):
    def __init__(self, base, witnesses=WITNESSES, witness_dropout=.2):
        super().__init__()
        if not 0 <= witness_dropout < 1 or not witnesses:
            raise ValueError('Nonempty witnesses and dropout in [0,1) required')
        self.base = base
        self.witnesses = tuple(witnesses)
        self.names = tuple(w[0] for w in witnesses)
        if len(set(self.names)) != len(self.names):
            raise ValueError('Duplicate witness name')
        self.witness_dropout = witness_dropout
        self.project = nn.ModuleDict({name: nn.Sequential(nn.Linear(b-a+d-c,32),nn.LayerNorm(32))
                                      for name,a,b,c,d in witnesses})
        self.identity = nn.ParameterDict({name:nn.Parameter(torch.zeros(8)) for name in self.names})
        for value in self.identity.values(): nn.init.normal_(value,std=.02)
        self.query = nn.Parameter(torch.randn(40)*.02)
        self.context = nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.LayerNorm(64))
        self.severity = nn.Embedding(3,8)
        self.word_head = nn.Sequential(nn.Linear(40+64+8+2,128),nn.GELU(),nn.Linear(128,1))
        self.calibration = nn.Sequential(nn.Linear(22,32),nn.GELU(),nn.Linear(32,1))
        for head in (self.word_head,self.calibration):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward(self, acoustic, side, words, mask, severity, n_words, text,
                *, order=None, present=None):
        if words.shape[-1]!=119 or text.shape[-1]!=58 or side.shape[-1]!=30:
            raise ValueError('Expected audited 583593 feature schema 119/58/30')
        if mask.shape!=words.shape[:2] or not torch.equal(mask.sum(1),n_words) or (n_words<=0).any():
            raise ValueError('Every reference word must be pooled exactly once')
        order=self.names if order is None else tuple(order)
        if len(order)!=len(self.names) or set(order)!=set(self.names):
            raise ValueError('Order must permute witnesses without changing identity')
        batch,count=mask.shape
        available=torch.ones(batch,len(self.names),device=words.device,dtype=torch.bool) if present is None else present
        if available.shape!=(batch,len(self.names)) or available.dtype!=torch.bool or not available.any(1).all():
            raise ValueError('At least one named witness must be available for each item')
        available=available.clone()
        if self.training and self.witness_dropout:
            kept=available & (torch.rand(available.shape,device=words.device)>=self.witness_dropout)
            empty=~kept.any(1)
            kept[empty,available[empty].long().argmax(1)]=True
            available=kept
        columns={w[0]:w[1:] for w in self.witnesses}
        tokens=[]
        for name in order:
            a,b,c,d=columns[name]
            features=words[...,a:b]
            if d>c:features=torch.cat((features,text[:,None,c:d].expand(-1,count,-1)),-1)
            # Missing witnesses can carry NaNs; masking happens before projection.
            keep=mask & available[:,self.names.index(name),None]
            features=torch.where(keep[...,None],features,torch.zeros_like(features))
            tokens.append(torch.cat((self.project[name](features),self.identity[name].expand(batch,count,-1)),-1))
        tokens=torch.stack(tokens,-2)
        active=available[:,[self.names.index(name) for name in order]]
        scores=(tokens*self.query).sum(-1)/(40**.5)
        weights=scores.masked_fill(~active[:,None,:],float('-inf')).softmax(-1)
        pooled=(weights[...,None]*tokens).sum(-2)
        context=self.base.whisper_projection(acoustic)
        base_logit=self.base.regressor[:-1](torch.cat((context,side),-1)).squeeze(-1)
        base_logit=base_logit+self.calibration(text[:,:22]).squeeze(-1)
        position=(torch.arange(count,device=words.device)[None,:]+.5)/n_words[:,None]
        inverse=n_words.float().reciprocal()[:,None].expand(-1,count)
        x=torch.cat((pooled,self.context(context)[:,None,:].expand(-1,count,-1),
                     self.severity(severity)[:,None,:].expand(-1,count,-1),position[...,None],inverse[...,None]),-1)
        logits=base_logit[:,None]+self.word_head(x).squeeze(-1)
        probs=logits.sigmoid()
        prediction=torch.where(mask,probs,torch.zeros_like(probs)).sum(1)/n_words
        return dict(prediction=prediction,word_logits=logits,word_probs=probs,
                    witness_attention=weights,witness_order=order)
