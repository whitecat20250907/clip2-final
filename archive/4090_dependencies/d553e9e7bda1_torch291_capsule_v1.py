"""Fixed official Torch2.9.1+cu128 single-process audio inference capsule.
Public wheels are SHA256 checked in RAM. Only small immutable assets persist under own root.
Large ELF files are sealed anonymous memfd objects; symlinks and all caches stay under own root.
Not a shared-environment install, not an accuracy experiment, no VALID export.
"""
import os,sys,json,time,hashlib,datetime,traceback,urllib.request,urllib.parse,io,zipfile,stat,fcntl,subprocess,runpy,gc
from pathlib import Path
Q=Path(__file__).resolve().parent;ROOT=Q/'torch291_capsule_v1';SITE=ROOT/'site';FLAT=ROOT/'native_libs';CACHE=Q/'torch291_small_cache_v1'
assert ROOT.resolve().is_relative_to(Q.resolve()) and CACHE.resolve().is_relative_to(Q.resolve())
ROOT.mkdir(exist_ok=False);SITE.mkdir();FLAT.mkdir();CACHE.mkdir(exist_ok=True)
sys.dont_write_bytecode=True
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 p=ROOT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',RAYON_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 with (ROOT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def available():
 return int([x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0])*1024
def owned_path(base,name):
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 q=base/p;assert q.resolve().is_relative_to(base.resolve());return q
started=time.monotonic();fds=[];native_bytes=0;receipts=[];aliases={};asset_records=[]
cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
def download(entry):
 assert available()>16*1024**3,'Memory pressure; stop without touching peer jobs'
 url=entry['url'];size=int(entry['size']);assert 0<size<2*1024**3
 assert urllib.parse.urlparse(url).scheme=='https' and urllib.parse.urlparse(url).hostname in ['download-r2.pytorch.org','download.pytorch.org','files.pythonhosted.org']
 buf=io.BytesIO();got=0;attempt=0;mark=128*2**20
 while got<size:
  attempt+=1;assert attempt<=5,'Public wheel retry limit exhausted'
  try:
   req=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(got,size-1),'User-Agent':'CLIP2-pinned-runtime-fidelity'})
   with urllib.request.urlopen(req,timeout=45) as r:
    if r.status==206:assert r.headers.get('Content-Range')=='bytes %d-%d/%d'%(got,size-1,size)
    else:assert r.status==200 and got==0 and int(r.headers.get('Content-Length'))==size
    while got<size:
     chunk=r.read(min(2**20,size-got))
     if not chunk:raise IOError('Incomplete public wheel response')
     buf.write(chunk);got+=len(chunk)
     if got>=mark:print('PUBLIC_WHEEL_PROGRESS',entry['name'],got,size,flush=True);mark+=128*2**20
  except Exception as ex:
   print('PUBLIC_WHEEL_RETRY',entry['name'],attempt,got,type(ex).__name__,str(ex)[:200],flush=True);time.sleep(2)
 digest=hashlib.sha256(buf.getbuffer()).hexdigest();assert digest==entry['sha256'],entry['name']
 buf.seek(0);print('PUBLIC_WHEEL_SHA_VERIFIED',entry['name'],digest,flush=True);return buf
def extract(entry,buf):
 global native_bytes,cache_bytes
 count=0;small_bytes=0;ram_bytes=0
 with zipfile.ZipFile(buf) as z:
  for info in z.infolist():
   if info.is_dir():continue
   name=info.filename;p=Path(name)
   assert not p.is_absolute() and '..' not in p.parts and not stat.S_ISLNK(info.external_attr>>16)
   is_native=('.so' in p.name)
   is_asset=name.endswith(('.py','.pyi','.json','.yaml','.yml','.txt')) or '.dist-info/' in name or any(t in p.name.upper() for t in ['LICENSE','COPYING','NOTICE']) or name=='torch/bin/torch_shm_manager'
   if entry['name']=='triton' and name.endswith(('.h','.cuh','.bc','.c','.cu')):is_asset=True
   if not (is_native or is_asset):continue
   if name.startswith(('torch/include/','torch/test/')):continue
   target=owned_path(SITE,name);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists() and not target.is_symlink()
   if is_native and info.file_size>2*2**20:
    assert native_bytes+info.file_size<=12*1024**3,'Native RAM budget exceeded'
    fd=os.memfd_create(p.name,os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING);fds.append(fd);h=hashlib.sha256();written=0
    with z.open(info) as f:
     for chunk in iter(lambda:f.read(2**20),b''):
      h.update(chunk);view=memoryview(chunk)
      while view:
       n=os.write(fd,view);assert n>0;written+=n;view=view[n:]
    assert written==info.file_size;os.lseek(fd,0,os.SEEK_SET)
    fcntl.fcntl(fd,fcntl.F_ADD_SEALS,fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE)
    target.symlink_to('/proc/%d/fd/%d'%(os.getpid(),fd))
    native_bytes+=written;ram_bytes+=written;digest=h.hexdigest();storage='sealed_memfd'
   else:
    raw=z.read(info);digest=hashlib.sha256(raw).hexdigest();cp=owned_path(CACHE,name);cp.parent.mkdir(parents=True,exist_ok=True)
    if cp.exists():
     assert cp.is_file() and not cp.is_symlink() and sha(cp)==digest
    else:
     assert cache_bytes+len(raw)<=96*2**20,'Small persistent asset budget exceeded'
     with cp.open('xb') as f:f.write(raw)
     cp.chmod(0o555 if name=='torch/bin/torch_shm_manager' else 0o444);cache_bytes+=len(raw)
    os.link(cp,target);small_bytes+=len(raw);storage='own_immutable_small_asset_hardlink'
   if is_native and p.name.startswith('lib'):
    alias=FLAT/p.name
    if p.name in aliases:assert aliases[p.name]==digest,'Conflicting native SONAME basename '+p.name
    else:alias.symlink_to(target);aliases[p.name]=digest
   asset_records.append(dict(package=entry['name'],path=name,bytes=info.file_size,sha256=digest,storage=storage));count+=1
 receipts.append(dict(name=entry['name'],version=entry['version'],wheel_sha256=entry['sha256'],wheel_bytes=entry['size'],files=count,small_bytes=small_bytes,native_RAM_bytes=ram_bytes))
 print('CAPSULE_PACKAGE_READY',entry['name'],count,'native_RAM_MiB',round(ram_bytes/2**20,2),'cache_MiB',round(cache_bytes/2**20,2),flush=True)
save('launch.json',dict(PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),available_RAM_bytes=available(),persistent_asset_cap_MiB=96,native_RAM_cap_GiB=12,shared_env_modified=False))
try:
 assert sys.version_info[:2]==(3,10);assert available()>32*1024**3
 assert str(FLAT) in os.environ.get('LD_LIBRARY_PATH','').split(':'),'Set own native path before process startup'
 manifest=read(Q/'PUBLIC_TORCH291_CUDA128_DEPENDENCY_MANIFEST_V1.json');wheel=read(Q/'PUBLIC_TORCH291_CU128_PREFLIGHT_V1.json')
 assert manifest['torch_wheel_bytes']==900927048
 url,digest=wheel['wheel_url'].split('#sha256=');assert digest=='c8d670aa0be6fbecd2b0e7b7d514a104dbdefcc3786ca446cf0c3415043ea40a'
 entries=[dict(name='torch',version='2.9.1+cu128',url=url,size=900927048,sha256=digest)]+manifest['dependencies']
 save('protocol.json',dict(scope='Pinned original Torch2.9.1+CUDA12.8 inference-only source-fidelity diagnosis on existingRTX4090. OriginalPy3.12/NumPy2.5/5090 remain different; must pass unchanged referencefeature/fullscore gates.',manifest_sha256=sha(Q/'PUBLIC_TORCH291_CUDA128_DEPENDENCY_MANIFEST_V1.json'),script_sha256=sha(Path(__file__)),worker_sha256=sha(Q/'numeric_audio16_torch291_v1.py'),wheels=entries,training=False,VALID=False,large_native_storage='Sealed anonymous RAM fd objects, own per-process symlinks, no persisted wheels; small assets own read-only cache',no_shared_environment_writes=True))
 # Keep CUDA selection fixed before torch import; maximum2GiB allocator in a currently idle GPU.
 candidates=[]
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 for line in r.stdout.splitlines():
  i,free,util=map(int,line.split(','))
  if free>=8192 and util<=5:candidates.append((free,-i,i))
 assert candidates,'No suitably idleGPU; do not claim numerical failure'
 chosen=max(candidates)[2];os.environ['CUDA_VISIBLE_DEVICES']=str(chosen);os.environ['CLIP_FIXED_GPU']=str(chosen)
 for entry in entries:
  buf=download(entry);extract(entry,buf);buf.close();del buf;gc.collect()
 save('public_package_receipts.json',dict(packages=receipts,files=asset_records,native_RAM_bytes=native_bytes,persistent_asset_cache_bytes=cache_bytes,descriptors=len(fds)))
 sys.path.insert(0,str(SITE))
 import torch,triton,sympy
 assert Path(torch.__file__).resolve().is_relative_to(SITE.resolve())
 assert torch.__version__=='2.9.1+cu128' and torch.version.cuda=='12.8'
 assert triton.__version__=='3.5.1' and sympy.__version__=='1.14.0'
 assert torch.cuda.is_available()
 free,total=torch.cuda.mem_get_info(0);assert free>=4*1024**3,'GPU no longer has safe freecapacity'
 torch.cuda.set_per_process_memory_fraction(2*1024**3/total,0);torch.set_num_threads(2)
 with torch.inference_mode():
  x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16);v=x@x;assert bool(torch.isfinite(v).all()) and float(abs(v.float()-32).max())==0
 del x,v;torch.cuda.empty_cache()
 save('runtime_preflight.json',dict(status='pinned_Torch291_CUDA128_RAM_native_import_and_BF16_matmul_passed',torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),triton=triton.__version__,sympy=sympy.__version__,GPU=chosen,device_name=torch.cuda.get_device_name(0),native_RAM_bytes=native_bytes,persistent_asset_cache_bytes=cache_bytes,shared_env_modified=False,worker_started=False))
 print('TORCH291_CAPSULE_RUNTIME_PREFLIGHT_PASSED',torch.__version__,torch.version.cuda,flush=True)
 runpy.run_path(str(Q/'numeric_audio16_torch291_v1.py'),run_name='__main__')
 result=read(Q/'numeric_audio16_torch291_v1/report.json')
 save('report.json',dict(status='pinned_Torch291_RAM_capsule_and_fixed16_diagnostic_complete',runtime_preflight_sha256=sha(ROOT/'runtime_preflight.json'),numeric_diagnostic_sha256=sha(Q/'numeric_audio16_torch291_v1/report.json'),worker_status=result['status'],native_RAM_bytes=native_bytes,persistent_asset_cache_bytes=cache_bytes,seconds=time.monotonic()-started,training=False,VALID=False,shared_env_modified=False))
 (ROOT/'exitcode').write_text('0');print('TORCH291_CAPSULE_COMPLETE',flush=True)
except BaseException:
 traceback.print_exc();(ROOT/'exitcode').write_text('1');raise
