import os,sys,pathlib,json,hashlib,math,types,importlib.util
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent;H=ROOT.parent;Q=H/'qwen_audio_runtime_v1';MODEL=pathlib.Path('/home/lisiyi/models/Qwen3-ASR-0.6B')
os.environ.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR']:
 p=ROOT/'runtime'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
sys.path.insert(0,str(Q/'tf4576_site_v1'))
import numpy as np,torch,soundfile as sf,soxr,transformers
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
torch.set_float32_matmul_precision('highest')
import transformers.utils.import_utils as availability
assert 'torchvision' not in sys.modules;availability._torchvision_available=False
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def put(p,x):
 with pathlib.Path(p).open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)
def save(p,**x):
 with pathlib.Path(p).open('xb') as f:np.savez_compressed(f,**x)
def mod(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def imports():
 pkg=types.ModuleType('_own_encoder_phone_reference');pkg.__path__=[str(Q/'reference')];sys.modules[pkg.__name__]=pkg
 conf=mod(pkg.__name__+'.configuration_qwen3_asr',Q/'reference/configuration_qwen3_asr.py')
 native=mod(pkg.__name__+'.modeling_qwen3_asr',Q/'reference/modeling_qwen3_asr.py')
 return conf,native
def own_mask(hidden,cu):
 assert hidden.ndim==2 and int(cu[0])==0 and int(cu[-1])==len(hidden)
 pos=torch.arange(len(hidden),device=hidden.device)
 group=torch.bucketize(pos,cu[1:],right=True);same=group[:,None]==group[None,:]
 return torch.zeros((len(hidden),len(hidden)),device=hidden.device,dtype=hidden.dtype).masked_fill(~same,torch.finfo(hidden.dtype).min)[None,None]
def mask_hook(module,args,kwargs):
 h=args[0] if args else kwargs['hidden_states'];cu=args[1] if len(args)>1 else kwargs['cu_seqlens']
 return args,{**kwargs,'attention_mask':own_mask(h,cu)}
def frozen_hash(m):
 h=hashlib.sha256()
 for n,p in sorted(m.named_parameters()):
  if p.requires_grad:continue
  key=n.replace('.base.','.')
  h.update(key.encode()+b'\0');h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def load_tower():
 conf,native=imports();raw=read(MODEL/'config.json')['thinker_config']['audio_config']
 c=conf.Qwen3ASRAudioEncoderConfig(**raw);c._attn_implementation='eager';c.conv_chunksize=8
 with torch.device('meta'):tower=native.Qwen3ASRAudioEncoder(c)
 expected=set(tower.state_dict());state={}
 with safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as f:
  for key in f.keys():
   if key.startswith('thinker.audio_tower.'):state[key[len('thinker.audio_tower.'):]]=f.get_tensor(key).float()
 assert set(state)==expected
 tower.load_state_dict(state,strict=True,assign=True)
 tower.positional_embedding=native.SinusoidsPositionEmbedding(c.max_source_positions,c.d_model)
 assert all(not p.is_meta and p.dtype==torch.float32 for p in list(tower.parameters())+list(tower.buffers()))
 assert all(torch.equal(tower.state_dict()[k],v) for k,v in state.items())
 tower.eval().requires_grad_(False)
 for layer in tower.layers:layer.register_forward_pre_hook(mask_hook,with_kwargs=True)
 extractor=transformers.WhisperFeatureExtractor.from_pretrained(str(MODEL),local_files_only=True)
 return tower,extractor,native,len(state)
def prepare(case,extractor):
 assert sha(case['audio'])==case['audio_sha']
 a,sr=sf.read(case['audio'],dtype='float32',always_2d=True);assert a.shape[1]==2 and np.isfinite(a).all()
 wave=a.mean(1,dtype=np.float32)
 if sr!=16000:wave=soxr.resample(wave,sr,16000,quality='HQ').astype(np.float32)
 original=len(wave);pad=(-original)%160;wave=np.pad(wave,(0,pad))
 z=extractor(wave,sampling_rate=16000,padding='longest',truncation=False,return_attention_mask=True,return_tensors='pt')
 mel=z['input_features'][0];length=int(z['attention_mask'][0].sum())
 assert mel.shape==(128,length) and length==len(wave)//160 and torch.isfinite(mel).all()
 return mel,dict(original_samples16k=original,right_zero_padding_samples=pad,mel_frames=length)
def forward(tower,mel):
 return tower(mel,feature_lens=torch.tensor([mel.shape[1]],device=mel.device,dtype=torch.long)).last_hidden_state
class LoRA(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.A=nn.Parameter(torch.empty(4,base.in_features,device=base.weight.device));self.B=nn.Parameter(torch.zeros(base.out_features,4,device=base.weight.device))
  nn.init.normal_(self.A,std=.02)
 def forward(self,x):return self.base(x)+2*F.linear(F.linear(x,self.A),self.B)
def insert_lora(tower):
 mods={}
 for i in range(14,18):
  for name in ['q_proj','v_proj']:
   module=LoRA(getattr(tower.layers[i].self_attn,name));setattr(tower.layers[i].self_attn,name,module);mods['layer%d.%s'%(i,name)]=module
 assert sum(p.numel() for m in mods.values() for p in [m.A,m.B])==57344
 return mods
def ctc_logp(head,frames,return_logits=False):
 logits=head(frames).double().cpu()
 up=F.interpolate(logits.T[None],scale_factor=2,mode='linear',align_corners=False)[0].T
 lp=up.log_softmax(-1)
 return (lp,logits) if return_logits else lp
def ctc(lp,seq):
 assert lp.ndim==2 and lp.shape[1]==40 and all(type(v) is int and 1<=v<=39 for v in seq)
 need=len(seq)+sum(a==b for a,b in zip(seq,seq[1:]));assert len(lp)>=need
 y=torch.tensor(seq,dtype=torch.long);tl=torch.tensor([len(seq)]);length=torch.tensor([len(lp)])
 val=F.ctc_loss(lp[:,None,:],y,length,tl,blank=0,reduction='none',zero_infinity=False)[0]
 assert torch.isfinite(val)
 return val
