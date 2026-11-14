
"""Word-level additive-logit branch; shared by fixed arms."""
import torch
from torch import nn
from torch.nn import functional as F
class NativeWordScore(nn.Module):
 def __init__(self,seed,dispersion):
  super().__init__()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed)
   self.branch=nn.Sequential(nn.Linear(339,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
   nn.init.zeros_(self.branch[-1].weight);nn.init.zeros_(self.branch[-1].bias)
  value=float(dispersion)-.001
  assert value>0
  self.eta=nn.Parameter(torch.tensor(value if value>20 else __import__('math').log(__import__('math').expm1(value)),dtype=torch.float32))
 def concentration(self):return F.softplus(self.eta)+.001
 def forward(self,x,parent,words):
  d=self.branch(x).squeeze(-1);z=torch.logit(parent.clamp(1e-7,1-1e-7))+d
  wp=z.sigmoid()*words
  return wp.sum(1)/words.sum(1),wp,z
def masked_word_bce(z,targets,mask):
 per=F.binary_cross_entropy_with_logits(z,targets,reduction='none')
 n=mask.sum(1);valid=n>0
 return ((per*mask).sum(1)/n.clamp_min(1))[valid].mean() if bool(valid.any()) else z.sum()*0
def independent_word_forward(state,x,parent,words):
 # Explicit functional readout, no module forward.
 h=F.gelu(F.linear(x,state['branch.0.weight'],state['branch.0.bias']),approximate='none')
 h=F.gelu(F.linear(h,state['branch.2.weight'],state['branch.2.bias']),approximate='none')
 d=F.linear(h,state['branch.4.weight'],state['branch.4.bias']).squeeze(-1)
 z=torch.log(parent.clamp(1e-7,1-1e-7))-torch.log1p(-parent.clamp(1e-7,1-1e-7))+d
 wp=z.sigmoid()*words
 return wp.sum(1)/words.sum(1),wp
def synthetic_check():
 torch.manual_seed(20261016);torch.set_num_threads(2)
 x=torch.randn(9,12,339);p=torch.rand(9,12);p[0,0]=0.;p[0,1]=1.
 words=torch.arange(12)[None,:]<torch.tensor([12,7,3,9,1,5,8,11,4])[:,None]
 model=NativeWordScore(44,1.8);other=NativeWordScore(44,1.8)
 assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
 q,w,z=model(x,p,words);gap=float(abs(w-p*words).max());assert gap<=2e-6
 nq,nw=independent_word_forward(model.state_dict(),x,p,words);assert abs(nw-w).max()<=2e-6
 pad=x.clone();pad[~words]=12345
 assert torch.equal(model(pad,p,words)[1],w)
 targets=torch.rand_like(p);mask=words.clone();mask[1]=False
 loss=masked_word_bce(z,targets,mask);opt=torch.optim.AdamW(model.parameters(),lr=.0003)
 loss.backward();g1=float(model.branch[-1].weight.grad.norm());assert g1>0
 opt.step();opt.zero_grad();q,w,z=model(x,p,words);masked_word_bce(z,targets,mask).backward()
 g2=float(model.branch[0].weight.grad.norm());assert g2>0 and all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
 empty=masked_word_bce(z,targets,torch.zeros_like(mask));assert float(empty)==0
 return dict(status='synthetic_zero_init_padding_ambiguous_mask_and_actual_gradients_passed',initial_maxgap=gap,output_gradient_step1=g1,hidden_gradient_step2=g2,real_training_updates=0,real_labels_read=False)
