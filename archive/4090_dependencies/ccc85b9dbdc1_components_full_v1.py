import os,sys,pathlib,json,hashlib,time
R=pathlib.Path(__file__).resolve().parent;G=R.parent;H=G.parent
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-d3964d16-3ea9-47fc-050c-0ed22aa087a9',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');sys.dont_write_bytecode=True
for k in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH']:
 p=R/'runtime'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
import numpy as np,torch,soundfile as sf,soxr,types
audio_source=pathlib.Path('/home/lisiyi/miniconda3/envs/icassp/lib/python3.10/site-packages/torchaudio/models/_hdemucs.py');hd=types.ModuleType('own_bound_hdemucs');sys.modules[hd.__name__]=hd;exec(compile(audio_source.read_text(),str(audio_source),'exec'),hd.__dict__)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
start=time.monotonic()
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def put(n,j):
 with(R/n).open('x')as f:json.dump(j,f,indent=2,allow_nan=False)
def note(**v):print('VOCAL_MASKING',json.dumps(dict(seconds=time.monotonic()-start,**v)),flush=True)
C=read(G/'PATHS_V1.json');hp=read(read(G/'PLAN_V1.json')['source_full_plan']);bp=read(hp['native_full_plan']);assets=read(bp['word_public'])
with np.load(read(G/'PLAN_V1.json')['micro_roles'],allow_pickle=False)as z:fit=z['FIT'].tolist()
free,total=torch.cuda.mem_get_info();assert free>4*1024**3;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/total);file=pathlib.Path('/home/lisiyi/.cache/torch/hub/torchaudio/models/hdemucs_high_trained.pt');model=hd.hdemucs_high(sources=['drums','bass','other','vocals']);model.load_state_dict(torch.load(file,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False).to('cuda');vi=list(model.sources).index('vocals');put('FROZEN_SEPARATOR_LOADED_FULL_V1.json',dict(weights_sha=sha(file),sources=list(model.sources),parameters=sum(p.numel()for p in model.parameters()),model='torchaudioHDEMUCS_HIGH_MUSDB_PLUS',torch=torch.__version__,torchaudio_source_sha=sha(audio_source),precision='FP32weights CUDAautocastFP16 conv/LSTM, float32output/STFTstats',chunk_seconds=4,overlap_seconds=.5,cap_GiB=2.5))
@torch.no_grad()
def separate(x):
 # New measurement uses short native chunks; no RMS equalization of the vocal stem.
 T=x.shape[-1];out=torch.zeros(4,2,T);den=torch.zeros(T);seg=4*44100;hop=int(3.5*44100);overlap=int(.5*44100);lo=0
 while lo<T:
  hi=min(lo+seg,T);u=x[:,lo:hi];length=len(u[0]);u=torch.nn.functional.pad(u,(0,max(0,44100-length)))
  with torch.autocast('cuda',dtype=torch.float16):v=model(u[None].to('cuda'))[0,:,:,:length].float().cpu()
  w=torch.ones(length);fade=min(overlap,length)
  if lo>0:w[:fade]=torch.linspace(0,1,fade)
  if hi<T:w[-fade:]=torch.linspace(1,0,fade)
  out[:,:,lo:hi]+=v*w;den[lo:hi]+=w
  if hi==T:break
  lo+=hop
 assert den.min()>0;return out/den
def measure(x,stems,qualify=False):
 v=stems[vi];m=x-v;components=torch.stack([v,m,x]);window=torch.hann_window(2048);z=torch.stft(components.reshape(6,-1),2048,hop_length=441,window=window,center=True,pad_mode='reflect',return_complex=True).reshape(3,2,1025,-1);pow=z.abs().square().double().numpy();w=np.ones(1025);w[1:-1]=2;pow*=w[None,None,:,None]/(2048*float(window.double().square().sum()));freq=np.fft.rfftfreq(2048,1/44100);bounds=[80,250,500,1000,2000,4000,8000];P=np.stack([pow[:,:,((freq>=l)&(freq<r))].sum(2)for l,r in zip(bounds[:-1],bounds[1:])],-1);V,M,X=P;eps=1e-12;vm=V.mean(0);mm=M.mean(0);xx=X.mean(0);sir=np.clip(np.log10((vm+eps)/(mm+eps)),-3,3);vf=vm/(vm+mm+eps);relative=np.log10((vm+eps)/(pow[2].sum(1).mean(0)[:,None]+eps));ears=np.log10((V+eps)/(M+eps));better=ears.max(0)-ears.mean(0);tot=pow.sum(2).mean(1);recon_error=((stems.sum(0)-x).double().square().mean()/x.double().square().mean()).sqrt();global_f=np.stack([np.clip(np.log10((tot[0]+eps)/(tot[1]+eps)),-3,3),tot[0]/(tot[0]+tot[1]+eps),np.log10((tot[0]+eps)/(tot[2]+eps)),np.full(tot.shape[1],float(recon_error))],-1);f=np.c_[sir,vf,relative,better,global_f];assert f.shape[1]==28 and np.isfinite(f).all();position=np.arange(len(f))*441/x.shape[-1];profile=np.stack([np.interp((np.arange(64)+.5)/64,position,f[:,j])for j in range(28)],-1);proof={}
 if qualify:
  n=x.shape[-1];frame=min(50,n//882);center=frame*441;xx0=np.pad(v[0].numpy(),(1024,1024),mode='reflect')[center:center+2048];ft=np.fft.rfft(xx0.astype(float)*window.numpy().astype(float));reference=abs(ft)**2*w/(2048*float(window.double().square().sum()));gap=float(abs(reference-pow[0,0,:,frame]).max());totalgap=abs(reference.sum()-float((torch.tensor(xx0).double()*window.double()).square().sum()/window.double().square().sum()));assert gap<1e-5 and totalgap<1e-10;proof=dict(independentFFT_power_maxgap=gap,Parseval_gap=totalgap)
 return profile,dict(reconstruction_residual_relativeRMS=float(recon_error),**proof)
roles=read(R/'MASKING_PUBLIC_ROLES_V1.json');ss=roles['signals'];rows=[];pending=[];receipts=read(R/'MASK_COMPONENT_RECEIPTS_V1.json')['receipts'];done=set(s for rec in receipts for s in rec['signals']);loaded_separator_sha=read(R/'FROZEN_SEPARATOR_LOADED_FULL_V1.json')['weights_sha']
def flush():
 global pending
 if not pending:return
 offset=np.r_[0,np.cumsum([len(a['mix'])for a in pending])];name='MASK_COMPONENTS_%03d_V1.npz'%len(receipts);tmp=R/(name+'.tmp');ids=[a['signal']for a in pending]
 with tmp.open('xb')as f:np.savez(f,signals=np.asarray(ids),offsets=offset,mix16=np.concatenate([a['mix']for a in pending]),vocal16=np.concatenate([a['vocal']for a in pending]))
 os.replace(tmp,R/name);rec=dict(file=name,sha=sha(R/name),signals=ids);receipts.append(rec);temp=R/'MASK_COMPONENT_RECEIPTS_V1.tmp';temp.write_text(json.dumps(dict(receipts=receipts,completed=sum(len(a['signals'])for a in receipts),total=len(ss),source_separator_sha=loaded_separator_sha,precision='frozenFP32weights AMP16convLSTM'),indent=2));os.replace(temp,R/'MASK_COMPONENT_RECEIPTS_V1.json');pending=[]
for j,s in enumerate(ss):
 if s in done:continue
 t=time.monotonic();f=assets[s]['files']['audio1'];assert sha(f['path'])==f['sha256'];wave,sr=sf.read(f['path'],dtype='float32',always_2d=True);assert wave.shape[1]==2 and np.isfinite(wave).all();raw=wave.mean(1,dtype=np.float32)
 if sr!=16000:raw=soxr.resample(raw,sr,16000,quality='HQ').astype(np.float32)
 if sr!=44100:wave=soxr.resample(wave,sr,44100,quality='HQ').astype(np.float32)
 x=torch.from_numpy(wave.T.copy());mu=x.mean();scale=x.mean(0).std().clamp_min(1e-8);normal=(x-mu)/scale;stems=separate(normal);vocal=(stems[vi]*scale).mean(0).numpy();vocal=soxr.resample(vocal,44100,16000,quality='HQ').astype(np.float32);assert len(vocal)==len(raw)and np.isfinite(vocal).all();pending.append(dict(signal=s,mix=raw,vocal=vocal));rows.append(dict(signal=s,seconds=time.monotonic()-t));done.add(s); 
 if len(pending)>=32:flush();note(phase='counterfactual_components',done=len(done),total=len(ss))
flush();assert sum(len(r['signals'])for r in receipts)==len(ss);put('MASK_COMPONENTS_COMPLETE_V1.json',dict(receipts=receipts,rows=rows,total=len(ss),roles_sha=sha(R/'MASKING_PUBLIC_ROLES_V1.json'),new_ASR_updates=0,human_labels_read=False));note(phase='components_complete')
