import pathlib,subprocess,json,time
r=pathlib.Path(__file__).resolve().parent
with(r/'PROSPECTIVE_ROLES_WORKER_LOG_V1.txt').open('xb')as f:
 c=subprocess.Popen(['/home/lisiyi/miniconda3/envs/icassp/bin/python','-u','-B',str(r/'prepare_prospective_confirmation_roles_v1.py')],stdout=f,stderr=subprocess.STDOUT)
 with(r/'PROSPECTIVE_ROLES_CHILD_V1.json').open('x')as g:json.dump(dict(pid=c.pid,start_ticks=pathlib.Path('/proc/'+str(c.pid)+'/stat').read_text().split()[21]),g)
 ec=c.wait()
with(r/'PROSPECTIVE_ROLES_TERMINAL_V1.json').open('x')as f:json.dump(dict(exitcode=ec,time=time.time()),f)
