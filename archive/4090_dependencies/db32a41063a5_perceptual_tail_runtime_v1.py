import os
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('CLIP2_GPU','GPU-28edfddb-06a3-d8c0-51c2-34bb0c8810d7'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from common_v1 import *
import copy,gc,math,torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
sys.path.insert(0,'/home/lisiyi/ICASSP2_2026/vendor/transformers_5_16_1')
import transformers
from transformers import AutoModelForCTC
from tokenizers import Tokenizer
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.enabled=False
free,total=torch.cuda.mem_get_info();assert free>6*1024**3;torch.cuda.set_per_process_memory_fraction(4*1024**3/total)
OLD=A/'Granite_last_block_qv_human_CTC_micro_v1/micro_train_v3_gpu7_matched_fresh_control';source_seal=read(OLD/'PUBLIC_PREFIX_CACHE_SEALED_BEFORE_TARGETS_V1.json')
prefixfile=OLD/'MICRO_PUBLIC_PENULTIMATE_F16_V1.npz';assert sha(prefixfile)==source_seal['cache_sha'];sm=nz(prefixfile,keys=['signals','offsets']);assert sm['signals'].tolist()==mids;off=sm['offsets']
with zipfile.ZipFile(prefixfile)as z:zi=z.getinfo('features.npy');assert zi.compress_type==zipfile.ZIP_STORED
with prefixfile.open('rb')as f:
 f.seek(zi.header_offset);h=f.read(30);nl,el=struct.unpack('<HH',h[26:30]);f.seek(zi.header_offset+30+nl+el);version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version);byteoffset=f.tell()
assert not fortran and dtype==np.dtype('float16')and shape[1]==1024
prefix=np.memmap(prefixfile,mode='r',dtype=dtype,shape=shape,offset=byteoffset)
S=G.parent/'under26_granite_native_vocabulary48_fastscreen_20261020_v1';vp=read(S/'VOCAB_PLAN_V1.json');snapshot=pathlib.Path(vp['snapshot'])
bound=read(A/'Granite_physical_stereo_native_v1/feasibility_v4_runtime_regex_dependency/RUNTIME_AND_INPUTS_BOUND_V1.json')
assert sha(snapshot/'model.safetensors')==bound['weights_sha']and sha(snapshot/'config.json')==bound['config_sha']
model,info=AutoModelForCTC.from_pretrained(str(snapshot),dtype=torch.float16,trust_remote_code=False,local_files_only=True,output_loading_info=True)
assert not any(info.get(k)for k in ['missing_keys','mismatched_keys','error_msgs']);model.eval().requires_grad_(False)
assert len(model.encoder.layers)==16 and 15 not in model.encoder.config.subsample_layers
base_layer=copy.deepcopy(model.encoder.layers[-1]).to('cuda').eval().requires_grad_(False);position=model.encoder.attention_dists.detach().clone().to('cuda');del model;gc.collect()
def frozen_sequence(i):return torch.tensor(np.asarray(prefix[off[i]:off[i+1]],np.float16),device='cuda')[None]
def tail(layer,i):return layer(frozen_sequence(i),position_embeddings=position)[0]
class QVLoRA(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.A=nn.Parameter(torch.randn(8,base.in_features,device='cuda',dtype=torch.float32)*.02);self.B=nn.Parameter(torch.zeros(base.out_features,8,device='cuda',dtype=torch.float32))
 def forward(self,x):return self.base(x)+(2*F.linear(F.linear(x.float(),self.A),self.B)).to(x.dtype)
def adapted_layer(seed=44):
 torch.manual_seed(20261022+seed);layer=copy.deepcopy(base_layer)
 for n in ['q_proj','v_proj']:setattr(layer.self_attn,n,QVLoRA(getattr(layer.self_attn,n)))
 layer.eval();assert sum(p.numel()for p in layer.parameters()if p.requires_grad)==32768
 return layer
