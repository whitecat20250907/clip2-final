"""Use own verified RAM keeper to run one frozen audio engineering worker."""
import os,sys,json,time,hashlib,pathlib,fcntl,subprocess,runpy,traceback,datetime,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--worker',required=True);ap.add_argument('--source-root');args=ap.parse_args()
Q=Path(__file__).resolve().parent;ROOT=Path(args.root);WORKER=Path(args.worker);SITE=ROOT/'site';FLAT=ROOT/'native_libs';SRC=Path(args.source_root) if args.source_root else Q/'torch291_RAM_keeper_v1'
assert ROOT.parent.resolve()==Q.resolve() and ROOT.is_dir() and set(ROOT.iterdir())=={FLAT}
assert WORKER.resolve().is_relative_to(Q.parent.resolve()) and WORKER.suffix=='.py'
sys.dont_write_bytecode=True;fds=[]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 with (ROOT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def available():
 return int([s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')][0])*1024
started=time.monotonic()
save('launch.json',dict(PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker=str(WORKER),worker_sha256=sha(WORKER),script_sha256=sha(Path(__file__)),training=False,VALID=False))
try:
 assert available()>32*1024**3 and str(FLAT) in os.environ.get('LD_LIBRARY_PATH','').split(':')
 assert SRC.parent.resolve()==Q.resolve() and SRC.name in ['torch291_RAM_keeper_v1','fp32_cache_resident_v1']
 if SRC.name=='torch291_RAM_keeper_v1':
  ready=read(SRC/'ready.json');pid=ready['PID'];assert ready['status']=='all41_verified_native_objects_shared_and_pinned_for_bounded_reuse' and pid==3193790
  cmd=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode();assert cmd.endswith(str(Q/'torch291_RAM_keeper_v1.py')+' ')
  native={r['path']:r for r in ready['native_objects']}
 else:
  ready=read(SRC/'launch.json');pid=ready['PID'];assert pid==3222897
  cmd=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
  assert str(Q/'own_resident_executor_v1.py') in cmd and str(Q.parent/'server_consistent_qwen_v1/produce_fp32_cache_v1.py') in cmd and '--root '+str(SRC) in cmd
  native={r['path']:r for r in read(SRC/'native_reuse.json')['objects']}
 assert len(native)==41
 SITE.mkdir();aliases={};verified=[]
 for base,dirs,files in os.walk(SRC/'site',followlinks=False):
  relbase=Path(base).relative_to(SRC/'site');dest=SITE/relbase;dest.mkdir(exist_ok=True,parents=True)
  assert not any((Path(base)/d).is_symlink() for d in dirs)
  for name in files:
   old=Path(base)/name;rel=old.relative_to(SRC/'site');new=SITE/rel
   if old.is_symlink():
    rec=native[str(rel)];target=os.readlink(old);assert target.startswith('/proc/%d/fd/'%pid)
    fd=os.open(target,os.O_RDONLY|os.O_CLOEXEC);fds.append(fd)
    assert os.fstat(fd).st_size==rec['bytes'] and fcntl.fcntl(fd,fcntl.F_GET_SEALS)&15==15
    h=hashlib.sha256()
    for b in iter(lambda:os.read(fd,2**20),b''):h.update(b)
    assert h.hexdigest()==rec['sha256'];os.lseek(fd,0,os.SEEK_SET);new.symlink_to('/proc/%d/fd/%d'%(os.getpid(),fd));verified.append(rec)
   else:
    assert old.is_file() and not old.stat().st_mode&0o222;os.link(old,new)
   if '.so' in name and name.startswith('lib'):
    if name in aliases:assert sha(new)==sha(aliases[name])
    else:(FLAT/name).symlink_to(new);aliases[name]=new
 assert len(verified)==41
 save('native_reuse.json',dict(objects=verified,large_native_bytes_persisted=0,large_native_RAM_objects_duplicated=False,keeper_modified=False))
 for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
  p=ROOT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',RAYON_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 deadline=time.monotonic()+1800;previous=None;stable=0;chosen=None
 while time.monotonic()<deadline:
  p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  candidates=[]
  for line in p.stdout.splitlines():
   i,free,util=map(int,line.split(','))
   if free>=6144 and util<=5:candidates.append((free,-i,i))
  current=previous if previous is not None and any(c[2]==previous for c in candidates) else (max(candidates)[2] if candidates else None)
  stable=stable+1 if current is not None and current==previous else (1 if current is not None else 0);previous=current
  if stable>=2:chosen=current;break
  assert available()>16*1024**3
  print('OWN_RESIDENT_WAIT_GPU',current,stable,flush=True);time.sleep(10)
 assert chosen is not None,'No safeidleGPU within30minutes; resourcewait, not numericalfailure'
 os.environ['CUDA_VISIBLE_DEVICES']=str(chosen);os.environ['CLIP_FIXED_GPU']=str(chosen)
 sys.path.insert(0,str(Q/'tf4576_site_v1'));sys.path.insert(0,str(SITE))
 import torch,transformers,huggingface_hub,triton,sympy
 assert Path(torch.__file__).resolve().is_relative_to(SITE.resolve()) and torch.__version__=='2.9.1+cu128'
 assert transformers.__version__=='4.57.6' and huggingface_hub.__version__=='0.36.0'
 assert torch.cuda.is_available()
 free,total=torch.cuda.mem_get_info(0);assert free>=6*1024**3
 torch.cuda.set_per_process_memory_fraction(2*1024**3/total,0);torch.set_num_threads(2)
 save('runtime_preflight.json',dict(status='same_own_original_versions_resident_reuse_passed',torch=torch.__version__,transformers=transformers.__version__,GPU=chosen,device_name=torch.cuda.get_device_name(0),allocator_cap_GiB=2,worker_sha256=sha(WORKER),shared_env_modified=False))
 print('RESIDENT_RUNTIME_READY',flush=True)
 runpy.run_path(str(WORKER),run_name='__main__')
 save('report.json',dict(status='worker_returned_successfully',worker=str(WORKER),seconds=time.monotonic()-started))
 (ROOT/'exitcode').write_text('0')
except BaseException:
 traceback.print_exc();(ROOT/'exitcode').write_text('1');raise
finally:
 for fd in fds:
  try:os.close(fd)
  except OSError:pass
