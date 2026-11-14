import pathlib,json,sys,time,ast,hashlib,os
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'PLAN_V1.json').read_text());C=json.loads((R/'PATHS_V1.json').read_text())
from paired_prefix_v1 import *
from paired_score_math_v1 import *
from scipy.special import rel_entr
p=read(R/'PLAN_V1.json')
def defs(file,names):
 s=pathlib.Path(file).read_text();nodes=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name in names];assert{n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),globals())
defs(p['source_study'],['public_hash','set_delta','features','disagreement','dual_features'])
def atomic(name,v):
 tmp=R/(name+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(R/name)
def loadwave(s):
 f=assets[s]['files']['audio1'];assert sha(f['path'])==f['sha256'];a,sr=sf.read(f['path'],dtype='float32',always_2d=True);assert a.shape[1]==2 and np.isfinite(a).all();w=a.mean(1,dtype=np.float32)
 if sr!=16000:w=soxr.resample(w,sr,16000,quality='HQ').astype(np.float32)
 assert len(w)>0;return w
def getmel(w):
 w=np.pad(w,(0,(-len(w))%160));z=extractor(w,sampling_rate=16000,padding='longest',truncation=False,return_attention_mask=True,return_tensors='pt');m=z['input_features'][0];nf=int(z['attention_mask'][0].sum());assert m.shape==(128,nf)and nf==len(w)//160;return m,nf
def variants(w,s):
 rms=float(np.sqrt(np.mean(w.astype(np.float64)**2)));seed=int(assets[s]['files']['audio1']['sha256'][:16],16);rng=np.random.default_rng(seed);noise=rng.standard_normal(len(w));noise-=noise.mean();noise/=max(float(np.sqrt(np.mean(noise**2))),1e-12);out={};checks={}
 for db in [30,20]:
  n=noise*rms*10**(-db/20);v=(w.astype(np.float64)+n).astype(np.float32);out['noise'+str(db)]=v
  actual=20*np.log10(rms/max(float(np.sqrt(np.mean((v.astype(np.float64)-w)**2))),1e-30))if rms>0 else float(db);assert abs(actual-db)<.001;checks['SNR'+str(db)]=float(actual)
 f=np.fft.rfftfreq(len(w),1/16000);t=np.clip((f-1500)/1000,0,1);mask=1-(1-10**(-6/20))*.5*(1-np.cos(np.pi*t));v=np.fft.irfft(np.fft.rfft(w)*mask,n=len(w));vrms=float(np.sqrt(np.mean(v**2)));v=v*(rms/vrms if vrms>0 else 1);out['band6']=v.astype(np.float32)
 assert all(a.shape==w.shape and np.isfinite(a).all()for a in out.values());checks['band_RMS_ratio']=float(np.sqrt(np.mean(out['band6'].astype(np.float64)**2))/rms)if rms>0 else 1.;assert abs(checks['band_RMS_ratio']-1)<1e-5
 return out,checks
start=time.monotonic();hp=read(p['source_full_plan']);bp=read(hp['native_full_plan']);assets=read(bp['word_public']);records=read(p['public_records']);roles=np.load(bp['roles'],allow_pickle=False);train=roles['strict_TRAIN_ids'].tolist();dev=roles['formal_DEV_ids'].tolist()
with np.load(p['micro_roles'],allow_pickle=False)as z:fit=z['FIT'].tolist();held=z['held'].tolist()
assert len(fit)==4096 and len(held)==1024 and set(fit)<=set(roles['FIT_ids'].tolist())and set(held)<=set(roles['historical_held_ids'].tolist())and not set(fit)&set(held)
ss=fit+held
with np.load(bp['qwen_readouts']['44'],allow_pickle=False)as z:
 ids=z['FIT_signals'].tolist()+z['held_signals'].tolist();values=np.r_[z['FIT_features'],z['held_features']];idx={s:i for i,s in enumerate(ids)};expected={s:values[idx[s]]for s in ss[:2]}
assert sha(p['asr_checkpoint'])==p['asr_checkpoint_sha']
seal=read(p['source_prefix_seal']);prefixExpected={}
for rec in seal['real_receipts']:
 wanted=set(ss[:2])&set(rec['signals'])
 if not wanted:continue
 with np.load(pathlib.Path(p['source_prefix_cache'])/rec['file'],allow_pickle=False)as z:
  ids=z['signals'].tolist();h=z['h14'];off=z['offsets']
  for s in wanted:
   j=ids.index(s);prefixExpected[s]=h[off[j]:off[j+1]].copy()
assert set(prefixExpected)==set(ss[:2])
free,total=torch.cuda.mem_get_info();assert free>p['min_free_GiB']*1024**3;torch.cuda.set_per_process_memory_fraction(p['cap_GiB']*1024**3/total)
tower,extractor,_,_=load_tower();tower.config._attn_implementation='sdpa';original=public_hash(tower);assert original==seal['frozen_public_parameter_sha'];mods=insert_lora(tower);cp=torch.load(p['asr_checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==945 and cp['seed']==44 and cp['public_frozen_parameter_sha']==original;set_delta(mods,cp['audio_delta'])
heads=nn.ModuleDict({k:nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,64),nn.GELU(),nn.Linear(64,40))for k in ['ref','response']});heads.load_state_dict(cp['heads']);tower.eval().requires_grad_(False).to('cuda:0');heads.eval().requires_grad_(False).to('cuda:0')
rng=np.random.default_rng(20261020);projections={d:torch.tensor(rng.standard_normal((d,64))/np.sqrt(d),dtype=torch.float32,device='cuda:0')for d in [896,1024]};capture=False;states={}
def hook(k):
 def call(module,args,kwargs):
  if capture:states[k]=(args[0]if args else kwargs['hidden_states']).detach()
 return call
handles=[tower.layers[k].register_forward_pre_hook(hook(k),with_kwargs=True,prepend=True)for k in [6,12,14]]
@torch.no_grad()
def run(m,keep=False):
 global capture,states
 states={};capture=keep
 with mathctx():fr=forward(tower,m.to('cuda:0'))
 capture=False;lp=np.stack([ctc_logp(heads[k],fr).numpy()for k in ['ref','response']],axis=1);out={}
 if keep:
  assert set(states)=={6,12,14};out['h14']=states[14].cpu().numpy().copy()
  for k,h in list(states.items())+[(18,fr)]:
   a=(h@projections[h.shape[-1]]).cpu().numpy().astype(np.float64);out['layer%d_128'%k]=np.r_[a.mean(0),a.std(0)]
 states={};return lp,out
nullbank={};memo={};proof=[];receipts=[];done=set();progressfile=R/'PERTURB_FEATURE_PROGRESS_V1.json'
if progressfile.exists():
 old=read(progressfile);receipts=old['receipts'];done=set(s for rec in receipts for s in rec['signals'])
 for rec in receipts:assert sha(R/rec['file'])==rec['sha256']
pending=[];count=len(done);cols=[0,3,4,5,6,7,8,9,10]
def flush():
 global pending
 if not pending:return
 fn='PERTURB_MICRO_FEATURES_CHUNK%03d_V1.npz'%len(receipts);keys=['stability72','original39','layer6_128','layer12_128','layer14_128','layer18_128'];data={k:np.asarray([r[k]for r in pending])for k in keys};ids=[r['signal']for r in pending];save(R/fn,signals=np.asarray(ids),**data);receipts.append(dict(file=fn,sha256=sha(R/fn),signals=ids));pending=[]
 atomic('PERTURB_FEATURE_PROGRESS_V1.json',dict(completed=count,total=len(ss),seconds=time.monotonic()-start,receipts=receipts))
with torch.no_grad():
 for pos,s in enumerate(ss):
  if s in done:continue
  wavsha=assets[s]['files']['audio1']['sha256'];w=loadwave(s);m,nf=getmel(w)
  if nf not in nullbank:nullbank[nf]=run(getmel(np.zeros(nf*160,np.float32))[0])[0]
  nlp=nullbank[nf];cachekey=wavsha
  if cachekey in memo:base,outs,pert=memo[cachekey]
  else:
   base,outs=run(m,True);vs,checks=variants(w,s);pert={}
   for name,wv in vs.items():pert[name]=run(getmel(wv)[0])[0]
   if len(memo)<8192:memo[cachekey]=(base,outs,pert)
  orig=dual_features(base,nlp,records[s]);values=[]
  for name in ['noise30','noise20','band6']:
   q=pert[name];feat=dual_features(q,nlp,records[s]);d=np.r_[feat[cols]-orig[cols],feat[np.asarray(cols)+18]-orig[np.asarray(cols)+18],disagreement(base[:,0],q[:,0]),disagreement(base[:,1],q[:,1])];assert d.shape==(24,);values.append(d)
  rec=dict(signal=s,original39=orig,stability72=np.concatenate(values),**{k:outs[k]for k in ['layer6_128','layer12_128','layer14_128','layer18_128']});pending.append(rec);count+=1
  if pos<2:
   srcm,_=prepare(dict(audio=assets[s]['files']['audio1']['path'],audio_sha=wavsha),extractor);assert torch.equal(srcm,m)
   assert np.array_equal(np.isnan(orig),np.isnan(expected[s]));gap=float(np.nanmax(abs(orig-expected[s])));pgap=float(abs(outs['h14']-prefixExpected[s]).max());assert gap<2e-5 and pgap<1e-6,(s,gap,pgap)
   dup,_=run(m);dgap=float(abs(dup-base).max());assert dgap==0.
   vs,checks=variants(w,s);proof.append(dict(signal=s,original39_maxgap=gap,prefix14_maxgap=pgap,duplicate_original_logP_gap=dgap,perturbation_checks=checks,mel_frames=nf))
   if pos==1:put(R/'PERTURB_FEATURE_QUALIFICATION_V1.json',dict(proof=proof,frozen_public_parameter_sha=original,ASR_checkpoint_sha=p['asr_checkpoint_sha'],new_ASR_updates=0,original_backend='FP32SDPA_MATH',input_perturbation_not_score_parameter_Jacobian=True))
  if count%32==0:print('PERTURB_FEATURES',count,len(ss),'seconds',round(time.monotonic()-start,1),flush=True)
  if len(pending)>=128:flush()
flush();assert count==len(ss)
parts={}
for rec in receipts:
 with np.load(R/rec['file'],allow_pickle=False)as z:
  for j,s in enumerate(z['signals'].tolist()):parts[s]={k:z[k][j].copy()for k in z.files if k!='signals'}
assert set(parts)==set(ss)
save(R/'PERTURB_MICRO_ALL_FEATURES_V1.npz',signals=np.asarray(ss),**{k:np.asarray([parts[s][k]for s in ss])for k in next(iter(parts.values()))})
put(R/'PERTURB_MICRO_FEATURES_COMPLETE_V1.json',dict(rows=count,feature_sha=sha(R/'PERTURB_MICRO_ALL_FEATURES_V1.npz'),seconds=time.monotonic()-start,ASR_updates=0,ASR_checkpoint_sha=p['asr_checkpoint_sha'],peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,receipts=receipts))
print('PERTURB_MICRO_FEATURES_COMPLETE',count,time.monotonic()-start,flush=True)
