"""Lease own41 immutable native runtime images without copying their RAM data."""
import os,sys,json,time,pathlib,hashlib,fcntl,traceback,datetime
Q=pathlib.Path(__file__).resolve().parent;SRC=Q/'strict24_prefix_runtime_v1';O=Q/'strict24_native_keeper_v1';SITE=O/'site';FLAT=O/'native_libs'
O.mkdir(exist_ok=False);SITE.mkdir();FLAT.mkdir();sys.dont_write_bytecode=True;fds=[]
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
start=time.monotonic()
save('launch.json',dict(PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),GPU=False,training=False))
try:
 source=read(SRC/'launch.json');pid=source['PID'];assert pid==1181711
 cmd=pathlib.Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
 assert str(Q/'own_resident_executor_v2.py') in cmd and 'produce_prefix_FP32_RAM_v1.py' in cmd and '--root '+str(SRC) in cmd
 assert read(SRC/'runtime_preflight.json')['status']=='same_own_original_versions_resident_reuse_passed'
 objects=read(SRC/'native_reuse.json')['objects'];native={r['path']:r for r in objects};assert len(native)==41
 records=[];aliases={};pure_count=0
 for base,dirs,files in os.walk(SRC/'site',followlinks=False):
  relative=pathlib.Path(base).relative_to(SRC/'site');dest=SITE/relative;dest.mkdir(parents=True,exist_ok=True)
  assert not any((pathlib.Path(base)/d).is_symlink() for d in dirs)
  for name in files:
   old=pathlib.Path(base)/name;rel=old.relative_to(SRC/'site');new=SITE/rel
   if old.is_symlink():
    rec=native[str(rel)];target=os.readlink(old);assert target.startswith('/proc/%d/fd/'%pid)
    source_stat=os.stat(target);fd=os.open(target,os.O_RDONLY|os.O_CLOEXEC);fds.append(fd)
    stat=os.fstat(fd);assert (stat.st_dev,stat.st_ino)==(source_stat.st_dev,source_stat.st_ino)
    assert stat.st_size==rec['bytes'] and fcntl.fcntl(fd,getattr(fcntl,'F_GET_SEALS',1034))&15==15
    h=hashlib.sha256()
    for block in iter(lambda:os.read(fd,2**20),b''):h.update(block)
    assert h.hexdigest()==rec['sha256'];os.lseek(fd,0,os.SEEK_SET)
    new.symlink_to('/proc/%d/fd/%d'%(os.getpid(),fd))
    records.append(dict(path=str(rel),bytes=rec['bytes'],sha256=rec['sha256'],fd=fd,source_same_inode=True))
   else:
    assert old.is_file() and not old.stat().st_mode&0o222
    os.link(old,new);pure_count+=1
   if '.so' in name and name.startswith('lib'):
    if name in aliases:assert sha(new)==sha(aliases[name])
    else:(FLAT/name).symlink_to(new);aliases[name]=new
 assert len(records)==41 and {x['path'] for x in records}==set(native)
 report=dict(status='owned_native41_readonly_handles_pinned_for8h_reuse',PID=os.getpid(),script_sha256=sha(__file__),source_root=str(SRC),source_PID=pid,source_native_manifest_sha256=sha(SRC/'native_reuse.json'),native_objects=objects,own_FD_records=records,native_bytes=sum(r['bytes'] for r in records),new_native_RAM_objects=0,new_native_disk_bytes=0,same_kernel_inodes_for_all41=True,pure_readonly_hardlinked_files=pure_count,source_modified=False,GPU=False,training=False,lease_seconds=8*3600,seconds=time.monotonic()-start)
 save('ready.json',report);print('OWN_NATIVE41_RUNTIME_LEASE_READY',json.dumps(dict(PID=os.getpid(),native_bytes=report['native_bytes'],new_native_RAM_objects=0,pure_readonly_files=pure_count)),flush=True)
 deadline=time.monotonic()+8*3600
 while time.monotonic()<deadline:time.sleep(20)
 save('lease_completed.json',dict(status='bounded8hleaseelapsed; duplicatedhandlesofactiveconsumersunaffected',seconds=time.monotonic()-start));(O/'exitcode').write_text('0')
except BaseException:
 traceback.print_exc();(O/'exitcode').write_text('1');raise
finally:
 for fd in fds:
  try:os.close(fd)
  except OSError:pass
