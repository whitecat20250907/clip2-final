import pathlib,json,time,subprocess
r=pathlib.Path(__file__).resolve().parent
with(r/'CACHE_AUDIT_WORKER_LOG_V1.txt').open('xb')as f:
 c=subprocess.Popen(['/home/lisiyi/miniconda3/envs/icassp/bin/python','-u','-B',str(r/'audit_existing_public_confirmation_sources_v1.py')],stdout=f,stderr=subprocess.STDOUT)
 with(r/'CACHE_AUDIT_CHILD_V1.json').open('x')as g:json.dump(dict(pid=c.pid,start_ticks=pathlib.Path('/proc/'+str(c.pid)+'/stat').read_text().split()[21]),g)
 ec=c.wait()
with(r/'CACHE_AUDIT_TERMINAL_V1.json').open('x')as f:json.dump(dict(exitcode=ec,time=time.time()),f)
