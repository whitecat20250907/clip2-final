"""Install pinned inference libraries in one owned directory; no shared writes."""
import pathlib, json, hashlib, urllib.request, io, zipfile, tarfile, stat, os, time, traceback, fcntl, threading
from concurrent.futures import ThreadPoolExecutor
R=pathlib.Path(__file__).resolve().parent
assert str(R).startswith('/home/lisiyi/zhaojiahao_1/runs/clip2_final_TEST2189_20261028/')
SITE=R/'site';SITE.mkdir(exist_ok=True);started=time.monotonic()
lock=(R/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
write_lock=threading.Lock();extract_lock=threading.Lock()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def put(n,v):
 with write_lock:
  p=R/(n+'.next');p.write_text(json.dumps(v,indent=2));p.replace(R/n)
def download(e):
 cached=R/'wheels'/e['url'].split('/')[-1]
 if cached.exists():
  assert cached.stat().st_size==e['size']and sha(cached)==e['sha256']
  return cached.open('rb')
 buf=io.BytesIO();size=e['size'];got=0
 for attempt in range(5):
  try:
   req=urllib.request.Request(e['url'],headers={'Range':'bytes=%d-%d'%(got,size-1),'User-Agent':'CLIP2-fixed-inference'})
   with urllib.request.urlopen(req,timeout=40)as f:
    assert f.status==206 or (f.status==200 and got==0)
    while got<size:
     b=f.read(min(2**20,size-got))
     if not b:raise IOError('Incomplete download')
     buf.write(b);got+=len(b)
     put('progress_'+e['name']+'.json',dict(package=e['name'],downloaded=got,total=size,seconds=time.monotonic()-started))
   break
  except Exception:
   if attempt==4:raise
   time.sleep(2)
 assert got==size and hashlib.sha256(buf.getbuffer()).hexdigest()==e['sha256'];buf.seek(0);return buf
try:
 cfg=json.loads((R/'protocol.json').read_text());entries=cfg['entries'];receipts=[]
 for name,version in [('peft','0.18.1'),('accelerate','1.12.0')]:
  with urllib.request.urlopen('https://pypi.org/pypi/'+name+'/'+version+'/json',timeout=30)as f:m=json.load(f)
  v=next(x for x in m['urls']if x['filename'].endswith('py3-none-any.whl'))
  entries.append(dict(name=name,version=version,url=v['url'],size=v['size'],sha256=v['digests']['sha256']))
 put('resolved_packages.json',entries)
 def install(e):
  receipt=R/('receipt_'+e['name']+'.json')
  if receipt.exists():
   old=json.loads(receipt.read_text());assert old['package']==e
   for x in old['files']:assert sha(SITE/x['path'])==x['SHA']
   return old
  buf=download(e);files=[]
  with extract_lock,zipfile.ZipFile(buf)as z:
   for i in z.infolist():
    if i.is_dir():continue
    p=pathlib.Path(i.filename);assert not p.is_absolute()and '..'not in p.parts and not stat.S_ISLNK(i.external_attr>>16)
    if i.filename.startswith(('torch/include/','torch/test/')):continue
    dst=SITE/p;dst.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
    with z.open(i)as src,dst.open('wb')as out:
     for b in iter(lambda:src.read(2**20),b''):h.update(b);out.write(b)
    if i.filename=='torch/bin/torch_shm_manager' or '/bin/'in i.filename:dst.chmod(0o755)
    files.append(dict(path=i.filename,SHA=h.hexdigest(),bytes=i.file_size))
  buf.close();rec=dict(package=e,files=files);put(receipt.name,rec);print('READY',e['name'],flush=True);return rec
 with ThreadPoolExecutor(max_workers=4)as pool:receipts=list(pool.map(install,entries))
 p=pathlib.Path(cfg['qwen_source']);assert sha(p)==cfg['qwen_source_SHA'];files=[]
 with tarfile.open(p,'r:gz')as t:
  for m in t.getmembers():
   p=pathlib.Path(m.name);assert not p.is_absolute() and '..'not in p.parts and not m.issym()and not m.islnk()
   if m.isdir():continue
   parts=p.parts;ix=parts.index('qwen_asr')if 'qwen_asr'in parts else None
   if ix is None:continue
   dst=SITE/pathlib.Path(*parts[ix:]);dst.parent.mkdir(parents=True,exist_ok=True);raw=t.extractfile(m).read();dst.write_bytes(raw);files.append(dict(path=str(dst.relative_to(SITE)),SHA=hashlib.sha256(raw).hexdigest()))
 assert (SITE/'qwen_asr/__init__.py').exists();put('qwen_source_receipt.json',files)
 put('complete.json',dict(packages=[x['package']for x in receipts],seconds=time.monotonic()-started,shared_environment_modified=False,training_updates=0))
except BaseException:
 put('attention.json',dict(trace=traceback.format_exc()));raise
