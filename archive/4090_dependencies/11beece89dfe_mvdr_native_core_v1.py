from sufficiency_native_v1 import *
from mvdr_operator_v1 import resynthesize,synthetic_qualification
from functools import lru_cache
CF=G/'counterfactual_masking_sensitivity_full_v1/MASK_COMPONENT_RECEIPTS_V1.json';RECEIPTS=read(CF);component_index={}
for j,rec in enumerate(RECEIPTS['receipts']):
 for s in rec['signals']:
  assert s not in component_index;component_index[s]=j
@lru_cache(maxsize=1)
def component_shard(j):
 rec=RECEIPTS['receipts'][j];p=CF.parent/rec['file'];assert sha(p)==rec['sha'];z=nz(p);assert z['signals'].tolist()==rec['signals'];return z
def component(s):
 z=component_shard(component_index[s]);j=z['signals'].tolist().index(s);a,b=z['offsets'][j:j+2];return z['mix16'][a:b].copy(),z['vocal16'][a:b].copy()
def stereo(s):
 item=assets[s]['files']['audio1'];raw=pathlib.Path(item['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'];x,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True);assert x.shape[1]==2
 if sr!=16000:x=soxr.resample(x,sr,16000,quality='HQ').astype(np.float32)
 return x
def profiles(s,prob):
 result=np.zeros((12,6))
 for w,word in enumerate(assets[s]['words']):
  win=locate(s,w,len(prob))['window'];assert len(win)==len(prob);result[w]=profile(prob,word,win)
 return result
def sym(a,b):return np.concatenate([(a+b)/2,abs(a-b)],-1)
def measure_mvdr(s,raw6,proof=False):
 x=stereo(s);mix,vocal=component(s);assert len(x)==len(mix)==len(vocal)
 mono=audio(s);assert float(abs(mono-mix).max())<=1e-7
 good,meta=resynthesize(x,mix,vocal);wrong,wmeta=resynthesize(x,mix,vocal,True)
 proofdata=dict(signal=s,samples=len(x),rawmono_component_maxgap=float(abs(mono-mix).max()),beam_meta=meta,wrong_meta=wmeta,beam_vs_wrong_relativeRMS=float(np.linalg.norm(good-wrong)/max(np.linalg.norm(good),1e-20)),ear_reference_RMS=np.sqrt(np.mean(x.astype(float)**2,axis=0)).tolist(),beam_RMS=np.sqrt(np.mean(good**2,axis=0)).tolist(),wrong_RMS=np.sqrt(np.mean(wrong**2,axis=0)).tolist())
 if proof:
  sw,_=resynthesize(x[:,::-1],mix,vocal);gap=float(abs(sw[:,::-1]-good).max());assert gap<=1e-8;proofdata['ear_swap_maxgap']=gap
  p0=probability_audio(mono);old=profiles(s,p0);gap=float(abs(old-raw6).max());assert gap<=.002;proofdata['old_mono6_replay_maxgap']=gap
 ps=[]
 for wave in [x[:,0],x[:,1],good[:,0],good[:,1],wrong[:,0],wrong[:,1]]:
  p=probability_audio(np.ascontiguousarray(wave,dtype=np.float32));ps.append(profiles(s,p));del p
 common=np.concatenate([raw6,np.zeros((12,12))],-1)
 vals=dict(common18=common,ears18=np.concatenate([raw6,sym(ps[0],ps[1])],-1),beam18=np.concatenate([raw6,sym(ps[2],ps[3])],-1),wrong18=np.concatenate([raw6,sym(ps[4],ps[5])],-1))
 for v in vals.values():v[len(assets[s]['words']):]=0;assert np.isfinite(v).all()
 proofdata['true_wrong_native_feature_maxgap']=float(abs(vals['beam18']-vals['wrong18']).max())
 return vals,proofdata
