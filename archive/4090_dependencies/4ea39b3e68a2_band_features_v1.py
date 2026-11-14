import pathlib,json,sys,time,ast,hashlib
R=pathlib.Path(__file__).resolve().parent
from paired_prefix_v1 import *
from paired_score_math_v1 import *
from scipy.special import rel_entr
p=read(R/'PLAN_V1.json');C=read(R/'PATHS_V1.json');start=time.monotonic()
def defs(file,names):
 nodes=[n for n in ast.parse(pathlib.Path(file).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert{n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),globals())
defs(p['source_study'],['public_hash','set_delta','features','disagreement'])
defs(p['qualified_micro_worker'],['loadwave','getmel','variants'])
def atomic(name,v):
 tmp=R/(name+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(R/name)
hp=read(p['source_full_plan']);bp=read(hp['native_full_plan']);assets=read(bp['word_public']);records=read(p['public_records']);seal=read(p['source_prefix_seal'])
with np.load(bp['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
ss=train+dev;assert len(train)==16180 and len(dev)==4059
with np.load(p['micro_features'],allow_pickle=False)as z:
 microids=z['signals'].tolist();microix={s:i for i,s in enumerate(microids)};microband=z['stability72'][:,48:72].copy()
assert len(microids)==5120 and len(set(microids))==5120 and set(microids)<=set(ss)
old39={}
with np.load(bp['qwen_readouts']['44'],allow_pickle=False)as z:
 ids=z['FIT_signals'].tolist()+z['held_signals'].tolist();x=np.r_[z['FIT_features'],z['held_features']]
 for j,s in enumerate(ids):old39[s]=x[j].copy()
with np.load(bp['qwen_DEV']['44'],allow_pickle=False)as z:
 for j,s in enumerate(z['signals'].tolist()):old39[s]=z['features'][j].copy()
assert set(old39)==set(ss)
free,total=torch.cuda.mem_get_info();assert free>p['min_free_GiB']*1024**3;torch.cuda.set_per_process_memory_fraction(p['cap_GiB']*1024**3/total)
tower,extractor,_,_=load_tower();tower.config._attn_implementation='sdpa';original=public_hash(tower);assert original==seal['frozen_public_parameter_sha'];assert sha(p['asr_checkpoint'])==p['asr_checkpoint_sha']
mods=insert_lora(tower);cp=torch.load(p['asr_checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==945 and cp['seed']==44 and cp['public_frozen_parameter_sha']==original;set_delta(mods,cp['audio_delta'])
heads=nn.ModuleDict({k:nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,64),nn.GELU(),nn.Linear(64,40))for k in ['ref','response']});heads.load_state_dict(cp['heads']);tower.eval().requires_grad_(False).to('cuda:0');heads.eval().requires_grad_(False).to('cuda:0')
cols=np.asarray([0,3,4,5,6,7,8,9,10])
def band(w):
 rms=float(np.sqrt(np.mean(w.astype(np.float64)**2)));f=np.fft.rfftfreq(len(w),1/16000);t=np.clip((f-1500)/1000,0,1);mask=1-(1-10**(-6/20))*.5*(1-np.cos(np.pi*t));v=np.fft.irfft(np.fft.rfft(w)*mask,n=len(w));vrms=float(np.sqrt(np.mean(v**2)));v=v*(rms/vrms if vrms>0 else 1);return v.astype(np.float32)
@torch.no_grad()
def fromcache(h,cu):
 with mathctx():fr=tail(tower,torch.tensor(h,device='cuda:0'),torch.tensor(cu,device='cuda:0'))
 return np.stack([ctc_logp(heads[k],fr).numpy()for k in ['ref','response']],axis=1)
@torch.no_grad()
def fromwave(w):
 m,_=getmel(w)
 with mathctx():fr=forward(tower,m.to('cuda:0'))
 return np.stack([ctc_logp(heads[k],fr).numpy()for k in ['ref','response']],axis=1)
def make24(s,base,q):
 # Dummy null=real: only the nine real-wave columns are read.
 a=features(q[:,0],q[:,0],records[s]);b=features(q[:,1],q[:,1],records[s]);orig=old39[s]
 return np.r_[a[cols]-orig[cols],b[cols]-orig[cols+18],disagreement(base[:,0],q[:,0]),disagreement(base[:,1],q[:,1])]
def cached(z,s):
 j=z['signals'].tolist().index(s);a,b=z['offsets'][j:j+2];ca,cb=z['cu_offsets'][j:j+2];return z['h14'][a:b].copy(),z['cu_values'][ca:cb].copy()
proof=[]
for s in microids[:2]:
 rec=next(q for q in seal['real_receipts']if s in q['signals'])
 with np.load(pathlib.Path(p['source_prefix_cache'])/rec['file'],allow_pickle=False)as z:h,cu=cached(z,s)
 w=loadwave(s);v=band(w);origv,_=variants(w,s);assert np.array_equal(v,origv['band6'])
 base=fromcache(h,cu);q=fromwave(v);feat=make24(s,base,q);expected=microband[microix[s]];assert np.array_equal(np.isnan(feat),np.isnan(expected));gap=float(np.nanmax(abs(feat-expected)));assert gap<2e-5,(s,gap)
 proof.append(dict(signal=s,feature24_maxgap=gap,band_wave_bitexact=True,source_prefix=True))
put(R/'BAND_QUALIFICATION_V1.json',dict(proof=proof,source_ASR_sha=p['asr_checkpoint_sha'],ASR_updates=0,backend='FP32_SDPA_MATH',cached_original=True,unused_null_summary_omitted=True))
assigned=[r for j,r in enumerate(seal['real_receipts'])if j%2==p['part']]
wanted=[s for rec in assigned for s in rec['signals']if s not in microix and s in old39];assert len(wanted)==len(set(wanted))
done=set();receipts=[]
if(R/'BAND_PROGRESS_V1.json').exists():
 for rec in read(R/'BAND_PROGRESS_V1.json')['receipts']:
  assert sha(R/rec['file'])==rec['sha256'];receipts.append(rec);done.update(rec['signals'])
assert done<=set(wanted);pending=[];count=len(done);memo={}
def flush():
 global pending
 if not pending:return
 fn='BAND_NEW_FEATURES_CHUNK%03d_V1.npz'%len(receipts);ids=[r[0]for r in pending];save(R/fn,signals=np.asarray(ids),band24=np.asarray([r[1]for r in pending]));receipts.append(dict(file=fn,sha256=sha(R/fn),signals=ids));pending=[]
 atomic('BAND_PROGRESS_V1.json',dict(completed=count,total=len(wanted),seconds=time.monotonic()-start,receipts=receipts))
for rec in assigned:
 use=[s for s in rec['signals']if s in set(wanted)and s not in done]
 if not use:continue
 with np.load(pathlib.Path(p['source_prefix_cache'])/rec['file'],allow_pickle=False)as source_z:
  z={k:source_z[k]for k in ['signals','h14','offsets','cu_offsets','cu_values']}
  for s in use:
   h,cu=cached(z,s);base=fromcache(h,cu);key=assets[s]['files']['audio1']['sha256']
   if key in memo:q=memo[key]
   else:
    q=fromwave(band(loadwave(s)))
    if len(memo)<8192:memo[key]=q
   feat=make24(s,base,q);assert feat.shape==(24,)and not np.isinf(feat).any();pending.append((s,feat));count+=1
   if count%64==0:print('BAND',p['part'],count,len(wanted),'seconds',round(time.monotonic()-start,1),flush=True)
   if len(pending)>=128:flush()
flush();assert count==len(wanted)
put(R/'BAND_PART_COMPLETE_V1.json',dict(rows=count,source_ASR_sha=p['asr_checkpoint_sha'],reused_micro=5120,seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,receipts=receipts,cache_receipts_trusted_pinned_seal=True,ASR_updates=0))
print('BAND_COMPLETE',p['part'],count,time.monotonic()-start,flush=True)
