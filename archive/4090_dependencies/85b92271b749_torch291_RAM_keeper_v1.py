"""Bounded keeper for this experiment's already verified immutable public native RAM.
Shares existing sealed objects, never copies native bytes to disk or runs a model.
"""
import os,sys,json,time,hashlib,fcntl,datetime,traceback
from pathlib import Path
Q=Path(__file__).resolve().parent;ROOT=Q/'torch291_RAM_keeper_v1';SRC=Q/'torch291_capsule_v4';SITE=ROOT/'site';FLAT=ROOT/'native_libs';PID=3186818
sys.dont_write_bytecode=True
assert ROOT.is_dir() and not list(ROOT.iterdir())
fds=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 with (ROOT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def available():
 return int([s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')][0])*1024
save('launch.json',dict(PID=os.getpid(),source_PID=PID,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Own alreadySHAverified sealed native RAM only, no model execution, no GPU, max1hour retention afterready',no_large_RAM_copy=True,no_large_disk_write=True))
try:
 assert available()>32*1024**3
 cmd=(Path('/proc/%d/cmdline'%PID)).read_bytes().replace(b'\0',b' ').decode()
 assert cmd.endswith(str(Q/'torch291_capsule_v4.py')+' ')
 assert sha(Q/'torch291_capsule_v4.py')=='be06561a7dfecb6cf7bf1f997dc0e912cc8f14dbb0542796d383e8612729ce66'
 receipt=json.loads((SRC/'public_package_receipts.json').read_text())
 assert len(receipt['packages'])==18 and receipt['native_RAM_bytes']==6632485174
 native={r['path']:r for r in receipt['files'] if r['storage']=='sealed_memfd'};assert len(native)==41
 SITE.mkdir();FLAT.mkdir();records=[];aliases={}
 for base,dirs,files in os.walk(SRC/'site',followlinks=False):
  relative=Path(base).relative_to(SRC/'site');dest=SITE/relative;dest.mkdir(exist_ok=True,parents=True)
  assert not any((Path(base)/d).is_symlink() for d in dirs)
  for name in files:
   old=Path(base)/name;rel=old.relative_to(SRC/'site');new=SITE/rel
   if old.is_symlink():
    rec=native[str(rel)];target=os.readlink(old);assert target.startswith('/proc/%d/fd/'%PID)
    fd=os.open(target,os.O_RDONLY|os.O_CLOEXEC);fds.append(fd)
    assert os.fstat(fd).st_size==rec['bytes']
    seals=fcntl.fcntl(fd,fcntl.F_GET_SEALS);assert seals&15==15
    h=hashlib.sha256()
    for b in iter(lambda:os.read(fd,2**20),b''):h.update(b)
    assert h.hexdigest()==rec['sha256'];os.lseek(fd,0,os.SEEK_SET)
    new.symlink_to('/proc/%d/fd/%d'%(os.getpid(),fd))
    records.append(dict(path=str(rel),bytes=rec['bytes'],sha256=h.hexdigest(),seals=seals))
   else:
    assert old.is_file() and not old.stat().st_mode&0o222
    os.link(old,new)
   if '.so' in name and name.startswith('lib'):
    if name in aliases:assert sha(new)==sha(aliases[name])
    else:(FLAT/name).symlink_to(new);aliases[name]=new
 assert len(records)==41 and sum(r['bytes'] for r in records)==6632485174
 ready=time.monotonic();save('ready.json',dict(status='all41_verified_native_objects_shared_and_pinned_for_bounded_reuse',PID=os.getpid(),source_PID=PID,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),expiry_after_seconds=3600,source_receipt_sha256=sha(SRC/'public_package_receipts.json'),native_objects=records,large_native_RAM_bytes_duplicated=0,large_native_bytes_persisted_to_disk=0,GPUs_used=0,source_process_modified=False))
 print('RAM_KEEPER_READY',os.getpid(),flush=True)
 reason='bounded1hour_retention_expired'
 while time.monotonic()-ready<3600:
  if available()<16*1024**3:reason='global_memory_pressure_release_own_references';break
  stop=ROOT/'release_request.json'
  if stop.exists():
   request=json.loads(stop.read_text());assert request['keeper_PID']==os.getpid()
   reason='own_explicit_release_request';break
  time.sleep(30)
 save('report.json',dict(status='own_RAM_keeper_terminal',reason=reason,seconds=time.monotonic()-ready,GPUs_used=0,shared_environment_modified=False))
 (ROOT/'exitcode').write_text('0')
except BaseException:
 traceback.print_exc();(ROOT/'exitcode').write_text('1');raise
finally:
 for fd in fds:
  try:os.close(fd)
  except OSError:pass
