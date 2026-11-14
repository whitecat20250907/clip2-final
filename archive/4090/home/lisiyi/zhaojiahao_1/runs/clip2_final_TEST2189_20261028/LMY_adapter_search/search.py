import os,pathlib,json,time,hashlib
r=pathlib.Path(__file__).parent;start=time.monotonic();hits=[];visited=0;errors=[]
for root in ['/home/lisiyi/clip2_lmy','/home/yangjing/workplace/clip2','/home/lisiyi/zhaojiahao_1']:
 for dp,ds,fs in os.walk(root,onerror=lambda e:errors.append(str(e))):
  ds[:]=[x for x in ds if x not in ['.git','node_modules','__pycache__']];visited+=1
  for name in fs:
   if ('adapter' in name.lower() and name.endswith(('.safetensors','.bin','.pt','.zip','.tar.gz'))) or '3322' in name or ('Qwen3-ASR-1.7B' in dp and name.endswith('.safetensors')):
    p=pathlib.Path(dp)/name
    try:
     sz=p.stat().st_size
     if name.endswith('.npy'):continue
     row=dict(path=str(p),bytes=sz)
     if 9000000<sz<11000000:row['SHA']=hashlib.sha256(p.read_bytes()).hexdigest()
     hits.append(row)
    except Exception as e:errors.append(str(e))
  if visited%500==0:(r/'progress.json').write_text(json.dumps(dict(directories=visited,matches=len(hits),seconds=time.monotonic()-start)))
(r/'complete.json').write_text(json.dumps(dict(directories=visited,hits=hits,errors=errors,seconds=time.monotonic()-start),indent=2))
