import os
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('CLIP2_GPU','GPU-d3964d16-3ea9-47fc-050c-0ed22aa087a9'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from common_v1 import *
for key,name in [('NUMBA_CACHE_DIR','numba'),('LIBROSA_CACHE_DIR','librosa'),('MPLCONFIGDIR','mpl')]:
 p=R/'runtime'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import copy,gc,math,torch,io,re
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
sys.path.insert(0,'/home/lisiyi/ICASSP2_2026/vendor/transformers_5_16_1')
from transformers import AutoModelForCTC,AutoProcessor
from word_event_core_v3 import *
from functools import lru_cache
from scipy.special import xlogy
import soundfile as sf,soxr
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.enabled=False
P=read(R/'NATIVE_CONFIGURATION_FROM_QUALIFIED_PUBLIC4_V1.json')
free,total=torch.cuda.mem_get_info();assert free>6*1024**3
torch.cuda.set_per_process_memory_fraction(4*1024**3/total)
put('RESOURCE_PREFLIGHT_V1.json',dict(time=time.time(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],free_GiB=free/1024**3,cap_GiB=4))
snapshot=pathlib.Path(P['snapshot'])
assert sha(snapshot/'config.json')==P['config_sha']
# The original immutable local model weight hash is already bound by the reused
# source qualification. Check the same actual weights once for this new runtime.
assert sha(snapshot/'model.safetensors')==P['model_weight_sha']
processor=AutoProcessor.from_pretrained(str(snapshot),trust_remote_code=False,local_files_only=True)
pm,info=AutoModelForCTC.from_pretrained(str(snapshot),dtype=torch.float16,trust_remote_code=False,local_files_only=True,output_loading_info=True)
assert not any(info.get(k)for k in ['missing_keys','mismatched_keys','error_msgs'])
assert len(pm.encoder.layers)==16
base_layer=copy.deepcopy(pm.encoder.layers[-1]).to('cuda').eval().requires_grad_(False)
position=pm.encoder.attention_dists.detach().clone().to('cuda')
pm.encoder.layers=pm.encoder.layers[:-1];pm.to('cuda').eval().requires_grad_(False)
assert sha(pathlib.Path(P['warm_checkpoint']))==P['warm_checkpoint_sha']
cp=torch.load(P['warm_checkpoint'],map_location='cpu',weights_only=False)
WA=cp['weights']['response.A'].float().to('cuda');WB=cp['weights']['response.B'].float().to('cuda');del cp
with safe_open(str(snapshot/'model.safetensors'),framework='pt',device='cpu')as fd:
 W=fd.get_tensor('encoder.out.weight').float().to('cuda');bias=fd.get_tensor('encoder.out.bias').float().to('cuda')
for q in [WA,WB,W,bias]:q.requires_grad_(False)
pieces=tokenizer_pieces(snapshot/'tokenizer.json');descriptions=[piece_description(p)for p in pieces]
assert len(pieces)==16384
@lru_cache(maxsize=2048)
def compiled(query):return compile_word(pieces,query,descriptions)
AP=pathlib.Path(P['alignment']);assert sha(AP)==P['alignment_sha'];al=nz(AP);assert al['signals'].tolist()==ss
def audio(s):
 item=assets[s]['files']['audio1'];raw=pathlib.Path(item['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 x,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True);x=x.mean(1,dtype=np.float32)
 if sr!=16000:x=soxr.resample(x,sr,16000,quality='HQ').astype(np.float32)
 assert x.ndim==1 and np.isfinite(x).all()
 return x
@torch.no_grad()
def probability_audio(x):
 d=processor(x,sampling_rate=16000,return_tensors='pt')
 h=pm.encoder(**{k:(v.to('cuda',dtype=torch.float16)if k=='input_features'else v.to('cuda'))for k,v in d.items()},output_hidden_states=False,return_dict=True).last_hidden_state[0]
 hidden=base_layer(h[None],position_embeddings=position)[0].float()
 z=F.linear(hidden,W,bias)+(hidden@WA.T)@WB.T
 prob=z.double().softmax(-1).cpu().numpy()
 assert prob.shape[1]==16384 and np.isfinite(prob).all()and abs(prob.sum(1)-1).max()<1e-12
 return prob
def profile(prob,query,window):
 model=compiled(query);pc=compress_prob(prob,model['classes'],model['C'])
 g=forward_event(pc,model['trans'],np.ones(len(prob),bool),model['final'],model['start'])
 l=forward_event(pc,model['trans'],window,model['final'],model['start'])
 assert -1e-9<=l<=g+1e-9 and g<=1+1e-9
 def vec(v):
  v=float(np.clip(v,0,1));return [v,float(logit(np.clip(v,1e-8,1-1e-8))),float(-xlogy(v,v)-xlogy(1-v,1-v))]
 return np.array(vec(g)+vec(l))
def locate(s,w,N):
 j=idx[s];a=al['weights'][al['offsets'][j]:al['offsets'][j+1],w].astype(np.float64);T=len(a);a=np.maximum(a,0)
 if a.sum()<=0:
  lo=0;hi=T-1;med=(T-1)/2;fallback=True
 else:
  c=np.cumsum(a)/a.sum();lo=max(0,int(np.searchsorted(c,.05))-1);hi=min(T-1,int(np.searchsorted(c,.95))+1);med=int(np.searchsorted(c,.5));fallback=False
 qwin=(np.arange(T)>=lo)&(np.arange(T)<=hi)
 return dict(lo=max(0,int(np.floor(lo*N/T))),hi=min(N,int(np.ceil((hi+1)*N/T))),center=int(round((med+.5)*N/T)),window=qwin,mass=a,fallback=fallback)
def crop_span(loc,N,margin):
 lo=max(0,loc['lo']-margin);hi=min(N,loc['hi']+margin)
 need=min(N,max(12800,hi-lo))
 lo=int(np.clip((lo+hi-need)//2,0,N-need));return lo,lo+need
def faded_crop(x,lo,hi):
 y=x[lo:hi].copy();assert len(y)==hi-lo
 if lo==0 and hi==len(x):return y
 edge=min(320,len(y)//2);ramp=(.5-.5*np.cos(np.pi*(np.arange(edge)+1)/(edge+1))).astype(np.float32)
 y[:edge]*=ramp;y[-edge:]*=ramp[::-1]
 return y
def wrong_span(x,lo,hi,energy):
 N=len(x);L=hi-lo;starts=np.unique(np.r_[np.arange(0,N-L+1,1280,dtype=int),N-L])
 good=(starts+L<=lo)|(starts>=hi)
 if good.any():starts=starts[good]
 Et=max(float(energy[hi]-energy[lo]),0);Es=np.maximum(energy[starts+L]-energy[starts],0)
 score=abs(np.log((Es+1e-20)/(Et+1e-20)));j=np.lexsort((starts,-abs(starts-lo),score))[0]
 wl=int(starts[j]);return wl,wl+L
def measure(s,proof=False):
 x=audio(s);N=len(x);p0=probability_audio(x);ep=np.r_[0.,np.cumsum(x.astype(np.float64)**2)]
 n=len(assets[s]['words']);common=np.zeros((12,20));target=common.copy();wrong=common.copy();records=[]
 for w,word in enumerate(assets[s]['words']):
  loc=locate(s,w,N);assert len(loc['window'])==len(p0)
  raw=profile(p0,word,loc['window']);geo=[];tv=[];wv=[];views=[]
  for margin in [2560,10240]:
   lo,hi=crop_span(loc,N,margin);wl,wh=wrong_span(x,lo,hi,ep)
   xx=faded_crop(x,lo,hi);yy=faded_crop(x,wl,wh);assert len(xx)==len(yy)>=min(12800,N)
   pp=probability_audio(xx);qq=probability_audio(yy)
   t=profile(pp,word,np.ones(len(pp),bool))[:3];v=profile(qq,word,np.ones(len(qq),bool))[:3]
   tv.extend(t-raw[:3]);wv.extend(v-raw[:3])
   framecenters=(np.arange(len(loc['mass']))+.5)*N/len(loc['mass']);den=loc['mass'].sum()
   mass=float(loc['mass'][(framecenters>=lo)&(framecenters<hi)].sum()/den)if den>0 else 0.
   overlap=max(0,min(hi,wh)-max(lo,wl))/(hi-lo)
   ratio=float(.5*np.log((float(yy.astype(float)@yy)+1e-20)/(float(xx.astype(float)@xx)+1e-20)))
   geo.extend([len(xx)/N,mass,overlap,ratio])
   views.append(dict(margin_samples=margin,target_span=[lo,hi],wrong_span=[wl,wh],frames=len(pp),word_event_target=float(t[0]),word_event_wrong=float(v[0]),overlap=overlap))
  common[w]=np.r_[raw,geo,np.zeros(6)];target[w]=np.r_[raw,geo,tv];wrong[w]=np.r_[raw,geo,wv]
  records.append(dict(word_index=w,word=word,alignment_fallback=loc['fallback'],views=views))
 assert all(np.isfinite(z).all()for z in [common,target,wrong])
 return common,target,wrong,records
