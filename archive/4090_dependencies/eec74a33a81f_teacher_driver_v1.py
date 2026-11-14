import os,sys,pathlib,json,subprocess,time
R=pathlib.Path(__file__).resolve().parent
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True)
def ticks(pid):return pathlib.Path('/proc/%d/stat'%pid).read_text().split()[21]
with(R/'TEACHER_WORKER_LOG_V1.txt').open('xb')as log:
 child=subprocess.Popen([sys.executable,'-u','-B',str(R/'teacher_worker_v1.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 put('TEACHER_WORKER_PID_V1.json',dict(pid=child.pid,start_ticks=ticks(child.pid)));code=child.wait()
put('TEACHER_DRIVER_TERMINAL_V1.json',dict(actual_exit_code=code,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
