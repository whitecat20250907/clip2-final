import pathlib,subprocess,json,time,os
r=pathlib.Path(__file__).resolve().parent
gpus=['GPU-d3964d16-3ea9-47fc-050c-0ed22aa087a9','GPU-28edfddb-06a3-d8c0-51c2-34bb0c8810d7','GPU-3b1bf515-4d29-d122-3a0e-7feb40b92e11']
workers=[]
for k,gpu in enumerate(gpus):
 env=os.environ.copy();env['CLIP2_GPU']=gpu;env['CLIP2_SHARD']=str(k)
 f=(r/('FEATURE_SHARD%d_LOG_V1.txt'%k)).open('xb')
 c=subprocess.Popen(['/home/lisiyi/miniconda3/envs/icassp/bin/python','-u','-B',str(r/'extract_local_dependence_shard_v1.py')],stdout=f,stderr=subprocess.STDOUT,env=env)
 with(r/('FEATURE_SHARD%d_CHILD_V1.json'%k)).open('x')as g:json.dump(dict(pid=c.pid,start_ticks=pathlib.Path('/proc/'+str(c.pid)+'/stat').read_text().split()[21],GPU_UUID=gpu,shard=k),g)
 workers.append((k,c,f))
codes={}
for k,c,f in workers:
 ec=c.wait();codes[str(k)]=ec;f.close()
 with(r/('FEATURE_SHARD%d_TERMINAL_V1.json'%k)).open('x')as f:json.dump(dict(exitcode=ec,time=time.time()),f)
if all(v==0 for v in codes.values()):
 with(r/'FEATURE_AGGREGATE_LOG_V1.txt').open('xb')as f:
  ec=subprocess.call(['/home/lisiyi/miniconda3/envs/icassp/bin/python','-u','-B',str(r/'aggregate_local_dependence_v1.py')],stdout=f,stderr=subprocess.STDOUT)
else:ec=1
with(r/'FEATURE_TERMINAL_V1.json').open('x')as f:json.dump(dict(exitcode=ec,worker_codes=codes,time=time.time()),f)
