"""Small independent-text/phone-query attention residual on frozen completeM2 logits."""
import math
import torch
from torch import nn
ARMS=('text_word_attention','text_hard_control','text_sentence_attention')
class AttentionResidual(nn.Module):
 def __init__(self,phone_vocab,arm):
  super().__init__()
  if arm not in ARMS:raise ValueError(arm)
  self.arm=arm;self.phone=nn.Embedding(phone_vocab,32,padding_idx=0);self.query=nn.Linear(64,64);self.key=nn.Linear(64,64);self.value=nn.Linear(64,64);self.norm=nn.LayerNorm(64);self.output=nn.Linear(64,1);self.raw_width=nn.Parameter(torch.tensor(0.))
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,frame,frame_mask,text,phone,spans,word_mask,base_pred,base_word):
  b,t,_=frame.shape;w=text.shape[1];present=phone!=0;ph=(self.phone(phone)*present[...,None]).sum(-2)/present.sum(-1,keepdim=True).clamp_min(1)
  q=self.query(torch.cat([text,ph],-1));positions=torch.arange(t,device=frame.device,dtype=frame.dtype);freq=torch.exp(torch.arange(0,64,2,device=frame.device,dtype=frame.dtype)*(-math.log(10000)/64))
  pos=torch.zeros(t,64,device=frame.device,dtype=frame.dtype);pos[:,0::2]=torch.sin(positions[:,None]*freq);pos[:,1::2]=torch.cos(positions[:,None]*freq)
  k=self.key(frame)+pos[None];v=self.value(frame)
  if self.arm=='text_sentence_attention':
   q=(q*word_mask[...,None]).sum(1,keepdim=True)/word_mask.sum(1)[:,None,None]
   score=torch.einsum('bwd,btd->bwt',q,k)/8
  elif self.arm=='text_hard_control':
   left=spans[...,0];right=torch.maximum(left+1,spans[...,1]);allowed=(positions[None,None,:]>=left[...,None])&(positions[None,None,:]<right[...,None])&frame_mask[:,None,:]
   if not allowed.any(-1)[word_mask].all():raise ValueError('Hard span outside real frames')
   score=torch.where(allowed,torch.zeros((),device=frame.device),torch.full((),-1e4,device=frame.device))
  else:
   center=spans.float().mean(-1);width=(spans[...,1]-spans[...,0]).float().clamp_min(6)/2
   width=width*(.5+self.raw_width.sigmoid())
   prior=-.5*((positions[None,None,:]-center[...,None])/width[...,None]).square()
   score=torch.einsum('bwd,btd->bwt',q,k)/8+prior
  score=score.masked_fill(~frame_mask[:,None,:],-1e4);att=score.softmax(-1);entropy=-(att*att.clamp_min(1e-12).log()).sum(-1);e=torch.einsum('bwt,btd->bwd',att,v)
  # The hard control includes the same query path, isolating learned soft frame weighting.
  delta=self.output(self.norm(e+q)).squeeze(-1)
  if delta.shape[1]==1:delta=delta.expand(-1,w);entropy=entropy.expand(-1,w)
  logits=torch.logit(base_word.clamp(1e-6,1-1e-6));change=torch.sigmoid(logits+delta)-torch.sigmoid(logits)
  pred=base_pred+(change*word_mask).sum(-1)/word_mask.sum(-1)
  if torch.any((pred < -1e-5)|(pred>1+1e-5)):raise ValueError('Attention probability range')
  return dict(prediction=pred.clamp(0,1),word_logits=logits+delta,word_probs=(base_word+change).clamp(0,1),attention_entropy=entropy,entropy_penalty=torch.relu(math.log(3)-entropy)[word_mask].mean())
