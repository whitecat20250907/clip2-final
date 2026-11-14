
import os,pathlib
here=pathlib.Path(__file__).resolve().parent
for key,name in [('NUMBA_CACHE_DIR','numba'),('LIBROSA_CACHE_DIR','librosa'),('MPLCONFIGDIR','mpl')]:
 p=here/'runtime'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
from perceptual_tail_runtime_v1 import *
from word_event_core_v3 import *
from functools import lru_cache
from scipy.special import xlogy
from transformers import AutoProcessor
import soundfile as sf,soxr,io,re
from local_intervention_core_v1 import local_energy_intervention,smooth_gate
PLAN=read(R/'LOCAL_DEPENDENCE_PROTOCOL_BEFORE_IMPLEMENTATION_V1.json')
assert sha(R/'local_intervention_core_v1.py')==PLAN['source_kernel_sha']
sp=read(R.parent/'deferred_CTC_score_risk_current_Count920_micro_v1/full_original4059_v1/FULL_CURRENT_COUNT920_CTC_RISK_PROTOCOL_BEFORE_FEATURES_AND_FITS_V1.json')
assert sha(sp['warm_checkpoint'])==sp['warm_checkpoint_sha']
cp=torch.load(sp['warm_checkpoint'],map_location='cpu',weights_only=False);WA=cp['weights']['response.A'].float().to('cuda');WB=cp['weights']['response.B'].float().to('cuda');del cp
with safe_open(str(snapshot/'model.safetensors'),framework='pt',device='cpu')as fd:W=fd.get_tensor('encoder.out.weight').float().to('cuda');bias=fd.get_tensor('encoder.out.bias').float().to('cuda')
for t in [WA,WB,W,bias]:t.requires_grad_(False)
pieces=tokenizer_pieces(snapshot/'tokenizer.json');descriptions=[piece_description(p)for p in pieces];assert len(pieces)==16384
@lru_cache(maxsize=2048)
def compiled(query):return compile_word(pieces,query,descriptions)
processor=AutoProcessor.from_pretrained(str(snapshot),trust_remote_code=False,local_files_only=True)
pm,info=AutoModelForCTC.from_pretrained(str(snapshot),dtype=torch.float16,trust_remote_code=False,local_files_only=True,output_loading_info=True)
assert not any(info.get(k)for k in ['missing_keys','mismatched_keys','error_msgs'])
pm.encoder.layers=pm.encoder.layers[:-1];assert len(pm.encoder.layers)==15 and pm.encoder.config.num_hidden_layers==16
pm.to('cuda').eval().requires_grad_(False)
AP=ROOT/'native_word_temporal_current_baseline_full_v1/ALL20239_PUBLIC_WORD_FRAME_ALIGNMENT_V1.npz';aps=read(AP.parent/'PUBLIC_ALIGNMENT_SEAL_BEFORE_TARGETS_V1.json');assert sha(AP)==aps['feature_sha']
al=nz(AP);assert al['signals'].tolist()==ss
def audio(s):
 item=assets[s]['files']['audio1'];b=pathlib.Path(item['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==item['sha256']
 a,sr=sf.read(io.BytesIO(b),dtype='float32',always_2d=True);x=a.mean(1,dtype=np.float32)
 if sr!=16000:x=soxr.resample(x,sr,16000,quality='HQ').astype(np.float32)
 return x
@torch.no_grad()
def prefix_audio(x):
 d=processor(x,sampling_rate=16000,return_tensors='pt')
 return pm.encoder(**{k:(v.to('cuda',dtype=torch.float16)if k=='input_features'else v.to('cuda'))for k,v in d.items()},output_hidden_states=False,return_dict=True).last_hidden_state[0]
@torch.no_grad()
def probability(h):
 hidden=base_layer(h[None],position_embeddings=position)[0].float()
 z=F.linear(hidden,W,bias)+(hidden@WA.T)@WB.T
 p=z.double().softmax(-1).cpu().numpy();assert abs(p.sum(1)-1).max()<1e-12
 return p
def profile(p,query,window):
 assert re.fullmatch("[a-z0-9']+",query)
 model=compiled(query);pc=compress_prob(p,model['classes'],model['C'])
 g=forward_event(pc,model['trans'],np.ones(len(p),bool),model['final'],model['start'])
 l=forward_event(pc,model['trans'],window,model['final'],model['start'])
 assert -1e-9<=l<=g+1e-9 and g<=1+1e-9
 def vec(v):
  v=float(np.clip(v,0,1));return[v,float(logit(np.clip(v,1e-8,1-1e-8))),float(-xlogy(v,v)-xlogy(1-v,1-v))]
 return np.array(vec(g)+vec(l))

for v in PLAN['review']:assert sha(v['path'])==v['sha']
def geometry(s,w,x,ep):
 N=len(x);j=idx[s];aa=al['weights'][al['offsets'][j]:al['offsets'][j+1],w].astype(np.float64);T=len(aa);assert np.isfinite(aa).all()and aa.min()>=-1e-10;aa=np.maximum(aa,0)
 if aa.sum()<=0:qlo=0;qhi=T-1;med=(T-1)/2;fallback=True
 else:
  c=np.cumsum(aa)/aa.sum();qlo=max(0,int(np.searchsorted(c,.05))-1);qhi=min(T-1,int(np.searchsorted(c,.95))+1);med=int(np.searchsorted(c,.5));fallback=False
 querywindow=(np.arange(T)>=qlo)&(np.arange(T)<=qhi)
 original_lo=max(0,int(np.floor(qlo*N/T)));original_hi=min(N,int(np.ceil((qhi+1)*N/T)))
 L=max(1,min(original_hi-original_lo,round(.640*16000),max(1,N//4)))
 center=int(round((med+.5)*N/T));lo=int(np.clip(center-L//2,0,N-L));hi=lo+L
 starts=np.unique(np.r_[np.arange(0,N-L+1,1280,dtype=int),N-L])
 valid=(starts+L<=lo)|(starts>=hi)
 candidates=starts[valid]
 if len(candidates)==0:candidates=starts
 energies=np.maximum(ep[candidates+L]-ep[candidates],0);Et=max(float(ep[hi]-ep[lo]),0);scores=abs(np.log((energies+1e-20)/(Et+1e-20)))
 order=np.lexsort((candidates,-abs(candidates-lo),scores));win=int(order[0]);wronglo=int(candidates[win]);wronghi=wronglo+L;Ew=float(energies[win])
 overlap=max(0,min(hi,wronghi)-max(lo,wronglo))/L
 if N>=4*L:assert overlap==0
 far=max([0,N-L],key=lambda q:abs(q-lo));Efar=max(float(ep[far+L]-ep[far]),0);farscore=abs(np.log((Efar+1e-20)/(Et+1e-20)))
 assert scores[win]<=farscore+1e-10
 centers=(np.arange(T)+.5)*N/T;mass=float(aa[(centers>=lo)&(centers<hi)].sum()/aa.sum())if aa.sum()>0 else 0.
 common=np.array([.5*np.log(max(Et/L,1e-20)),L/N,mass,np.log((Ew+1e-20)/(Et+1e-20)),abs(wronglo-lo)/N,float(fallback)])
 meta=dict(target_span=[lo,hi],wrong_span=[wronglo,wronghi],candidate_windows=len(candidates),control_energy_log_distance=float(scores[win]),old_faredge_energy_log_distance=float(farscore),wrong_target_overlap_fraction=overlap,public_occupancy_mass_covered=mass,public_fallback=fallback)
 return querywindow,lo,hi,wronglo,wronghi,common,meta
@torch.no_grad()
def measure(i,proof=False):
 s=mids[i];x=audio(s);h=torch.from_numpy(np.array(prefix[off[i]:off[i+1]],np.float16)).to('cuda')
 if proof:assert torch.equal(prefix_audio(x),h),'rawprefix_not_bitexact'
 p0=probability(h);xx=x.astype(np.float64);ep=np.r_[0.,np.cumsum(xx*xx)];E=float(xx@xx);assert E>0
 feat=np.zeros((12,26));legal=np.zeros(12,bool);diag=np.zeros((12,8));proofrows=[]
 for w,query in enumerate(assets[s]['words']):
  window,lo,hi,wronglo,wronghi,physical6,meta=geometry(s,w,x,ep);assert len(window)==len(h)
  noise=np.random.default_rng(int(hashlib.sha256((s+'|boundedlocal_v2|'+str(w)).encode()).hexdigest()[:16],16)).standard_normal(hi-lo)
  noop,_=local_energy_intervention(x,lo,hi,noise,0.);assert np.array_equal(noop,x)
  original=profile(p0,query,window);values={};amps={};rmsgaps=[]
  for arm,a,b in [('target',lo,hi),('wrong',wronglo,wronghi)]:
   y,amp=local_energy_intervention(x,a,b,noise);amps[arm]=amp
   oldseg=x[a:b].astype(np.float64);newseg=y[a:b].astype(np.float64);newE=E+float(newseg@newseg-oldseg@oldseg);gap=abs(np.sqrt(newE/E)-1);assert gap<=2e-7;rmsgaps.append(gap)
   hh=prefix_audio(y);assert len(hh)==len(h);values[arm]=profile(probability(hh),query,window)
  nvec=smooth_gate(len(noise))*noise;injected=amps['target']**2*float(nvec@nvec)/E
  physical8=np.r_[physical6,np.log((amps['wrong']+1e-20)/(amps['target']+1e-20)),injected]
  feat[w]=np.r_[original,values['target']-original,values['wrong']-original,physical8];legal[w]=True
  diag[w]=[meta['control_energy_log_distance'],meta['old_faredge_energy_log_distance'],meta['wrong_target_overlap_fraction'],max(rmsgaps),meta['public_occupancy_mass_covered'],amps['target'],amps['wrong'],meta['candidate_windows']]
  if proof:proofrows.append(dict(signal=s,word_index=w,**meta,RMS_maxgap=max(rmsgaps),target_global_delta=float(values['target'][0]-original[0]),target_local_delta=float(values['target'][3]-original[3])))
 assert np.isfinite(feat).all()and np.isfinite(diag).all()
 return feat,legal,diag,proofrows
