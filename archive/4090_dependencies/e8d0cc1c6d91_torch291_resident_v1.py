"""Reuse verified own sealed native RAM, with reviewed 2GiB GPU cap + >=4GiB reserve.
No package downloads, no shared environment changes, no accuracy label reads.
"""
import os,sys,json,time,hashlib,pathlib,fcntl,signal,subprocess,runpy,traceback,datetime
from pathlib import Path
Q=Path(__file__).resolve().parent
ROOT=Q/'torch291_resident_v1';FLAT=ROOT/'native_libs';SITE=ROOT/'site'
SRC=Q/'torch291_capsule_v3';SOURCE_PID=3152504
assert ROOT.is_dir() and set(ROOT.iterdir())=={FLAT} and FLAT.is_dir()
sys.dont_write_bytecode=True
assert str(FLAT) in os.environ.get('LD_LIBRARY_PATH','').split(':')
fds=[];source_stopped=False;source_retired=False
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
started=time.monotonic()
save('launch.json',dict(PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_PID=SOURCE_PID,source_script_sha256=sha(Q/'torch291_capsule_v3.py'),worker_sha256=sha(Q/'numeric_audio16_torch291_v1.py'),GPU_cap_GiB=2,GPU_minfree_GiB=6,required_idle_util_max=5,revision_scope='Resource admission only: retain2GiB allocator cap, require at least4GiB reserve instead of6GiB. Numerical thresholds, weights, audio and16 IDs unchanged.',training=False,VALID=False))
try:
 assert available()>32*1024**3
 proc=Path('/proc/%d'%SOURCE_PID)
 cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert cmd.endswith(str(Q/'torch291_capsule_v3.py')+' ') and 'python -B -u ' in cmd
 assert not (SRC/'runtime_preflight.json').exists() and not (Q/'numeric_audio16_torch291_v1').exists()
 assert 'WAITING_FOR_IDLE_GPU_AFTER_PUBLIC_RUNTIME_READY' in (Q/'torch291_capsule_v3_launch.log').read_text()[-500:]
 receipt=read(SRC/'public_package_receipts.json')
 assert len(receipt['packages'])==18 and receipt['native_RAM_bytes']==6632485174
 assert sha(Q/'torch291_capsule_v3.py')=='54f3c050792f1dc40ad629eb6a385d250e2f591157e398dbdc905e4b30273bb1'
 os.kill(SOURCE_PID,signal.SIGSTOP);source_stopped=True
 # Wait only for our known process to become stopped, not for any peer resource.
 for _ in range(20):
  status=(proc/'status').read_text()
  if any(l.startswith('State:') and ('T' in l or 't' in l) for l in status.splitlines()):break
  time.sleep(.1)
 else:raise RuntimeError('Own source process did not reach stopped state')
 native={r['path']:r for r in receipt['files'] if r['storage']=='sealed_memfd'}
 assert len(native)==41
 SITE.mkdir();aliases={};verified=[]
 for base,dirs,files in os.walk(SRC/'site',followlinks=False):
  relative=Path(base).relative_to(SRC/'site');dest=SITE/relative;dest.mkdir(exist_ok=True,parents=True)
  assert not any((Path(base)/d).is_symlink() for d in dirs)
  for name in files:
   old=Path(base)/name;rel=old.relative_to(SRC/'site');new=SITE/rel
   if old.is_symlink():
    rec=native[str(rel)];target=os.readlink(old)
    assert target.startswith('/proc/%d/fd/'%SOURCE_PID)
    fd=os.open(target,os.O_RDONLY|os.O_CLOEXEC);fds.append(fd)
    assert os.fstat(fd).st_size==rec['bytes']
    seals=fcntl.fcntl(fd,fcntl.F_GET_SEALS)
    assert seals & (fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE)==15
    h=hashlib.sha256()
    for b in iter(lambda:os.read(fd,2**20),b''):h.update(b)
    assert h.hexdigest()==rec['sha256'];os.lseek(fd,0,os.SEEK_SET)
    new.symlink_to('/proc/%d/fd/%d'%(os.getpid(),fd))
    verified.append(dict(path=str(rel),sha256=h.hexdigest(),bytes=rec['bytes'],seals=seals))
   else:
    assert old.is_file() and not old.stat().st_mode & 0o222
    os.link(old,new)
   if '.so' in name and name.startswith('lib'):
    if name in aliases:assert sha(new)==sha(aliases[name])
    else:(FLAT/name).symlink_to(new);aliases[name]=new
 assert len(verified)==41 and sum(r['bytes'] for r in verified)==receipt['native_RAM_bytes']
 save('resident_transfer_receipt.json',dict(source_receipt_sha256=sha(SRC/'public_package_receipts.json'),native_objects=verified,large_native_bytes_copied_to_disk=0,large_native_RAM_objects_copied=0,new_process_holds_own_open_descriptors=True,small_assets='Hardlinks to own immutable SHA-verified files',source_proc_was_our_exact_waiting_capsule=True))
 # New descriptors keep sealed native objects alive; retire only the old own waiting process.
 os.kill(SOURCE_PID,signal.SIGTERM);os.kill(SOURCE_PID,signal.SIGCONT)
 source_stopped=False;source_retired=True
 save('source_process_retirement.json',dict(PID=SOURCE_PID,reason='Verified all41 native descriptors into own successor before retirement. Resource-only admission reassessment; no source numeric worker had started.',numeric_failure=False,original_exit_marker_may_be_absent_due_operator_SIGTERM=True))
 for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
  p=ROOT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',RAYON_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 deadline=time.monotonic()+1800;previous=None;stable=0;chosen=None
 while time.monotonic()<deadline:
  r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  candidates=[]
  for line in r.stdout.splitlines():
   i,free,util=map(int,line.split(','))
   if free>=6144 and util<=5:candidates.append((free,-i,i))
  current=max(candidates)[2] if candidates else None
  stable=stable+1 if current is not None and current==previous else (1 if current is not None else 0)
  previous=current
  print('RESOURCE_ADMISSION_OBSERVATION',current,'stable',stable,flush=True)
  if stable>=2:chosen=current;break
  assert available()>16*1024**3
  time.sleep(10)
 assert chosen is not None,'No suitably idle GPU within30minutes; no numerical inference failure'
 os.environ['CUDA_VISIBLE_DEVICES']=str(chosen);os.environ['CLIP_FIXED_GPU']=str(chosen)
 sys.path.insert(0,str(SITE))
 import torch,triton,sympy
 assert Path(torch.__file__).resolve().is_relative_to(SITE.resolve())
 assert torch.__version__=='2.9.1+cu128' and torch.version.cuda=='12.8'
 assert triton.__version__=='3.5.1' and sympy.__version__=='1.14.0'
 assert torch.cuda.is_available()
 free,total=torch.cuda.mem_get_info(0);assert free>=6*1024**3
 torch.cuda.set_per_process_memory_fraction(2*1024**3/total,0);torch.set_num_threads(2)
 with torch.inference_mode():
  x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16);v=x@x
  assert bool(torch.isfinite(v).all()) and float(abs(v.float()-32).max())==0
 del x,v;torch.cuda.empty_cache()
 save('runtime_preflight.json',dict(status='pinned_Torch291_CUDA128_resident_native_import_and_BF16_matmul_passed',torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),triton=triton.__version__,sympy=sympy.__version__,GPU=chosen,device_name=torch.cuda.get_device_name(0),shared_environment_modified=False,resource_reserve_GiB=4,allocator_cap_GiB=2))
 print('RESIDENT_TORCH291_RUNTIME_PREFLIGHT_PASSED',torch.__version__,flush=True)
 runpy.run_path(str(Q/'numeric_audio16_torch291_v1.py'),run_name='__main__')
 result=read(Q/'numeric_audio16_torch291_v1/report.json')
 save('report.json',dict(status='pinned_Torch291_resident_capsule_and_fixed16_diagnostic_complete',worker_status=result['status'],diagnostic_report_sha256=sha(Q/'numeric_audio16_torch291_v1/report.json'),seconds=time.monotonic()-started,training=False,VALID=False,shared_environment_modified=False))
 (ROOT/'exitcode').write_text('0')
except BaseException:
 traceback.print_exc()
 if source_stopped and not source_retired:
  try:os.kill(SOURCE_PID,signal.SIGCONT)
  except ProcessLookupError:pass
 (ROOT/'exitcode').write_text('1')
 raise
