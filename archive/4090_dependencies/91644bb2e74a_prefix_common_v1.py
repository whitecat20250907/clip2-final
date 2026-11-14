from common_v1 import *
class PrefixCut(RuntimeError):pass
class Prefix:
 def __init__(self,tower):
  self.tower=tower;self.enabled=False;self.h=None;self.cu=None
  self.hook=tower.layers[14].register_forward_pre_hook(self.capture_hook,with_kwargs=True,prepend=True)
 def capture_hook(self,module,args,kwargs):
  if not self.enabled:return None
  h=args[0] if args else kwargs['hidden_states'];cu=args[1] if len(args)>1 else kwargs['cu_seqlens']
  self.h=h.detach().cpu().clone();self.cu=cu.detach().cpu().clone();raise PrefixCut()
 def __call__(self,mel):
  self.enabled=True;self.h=None;self.cu=None
  try:
   with torch.no_grad():forward(self.tower,mel)
  except PrefixCut:pass
  else:raise AssertionError('Prefix capture was not called')
  finally:self.enabled=False
  h,cu=self.h,self.cu;self.h=None;self.cu=None
  assert h is not None and h.ndim==2 and h.shape[1]==896 and h.dtype==torch.float32 and torch.isfinite(h).all()
  assert cu.dtype==torch.int32 and int(cu[0])==0 and int(cu[-1])==len(h) and bool((cu[1:]>cu[:-1]).all())
  return h,cu
def tail(tower,h,cu):
 for layer in tower.layers[14:]:h=layer(h,cu)[0]
 return tower.proj2(tower.act(tower.proj1(tower.ln_post(h))))
def mathctx():return torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
def check_binding():
 b=read(ROOT/'SOURCE_BINDING_V1.json')
 for n,h in b['files'].items():assert sha(ROOT/n)==h,n
 for p,h in b['upstream'].items():assert sha(p)==h,p
 return read(ROOT/'CACHE_PLAN_V1.json')
