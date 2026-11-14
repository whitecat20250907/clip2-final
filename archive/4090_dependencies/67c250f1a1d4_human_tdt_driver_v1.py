import os,sys,pathlib,json,subprocess,time
R=pathlib.Path(__file__).resolve().parent
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True)
def ticks(pid):return pathlib.Path('/proc/%d/stat'%pid).read_text().split()[21]
code=0
with(R/'HUMAN_TDT_WORKER_LOG_V1.txt').open('xb')as log:
 for phase in ['train','score']:
  child=subprocess.Popen([sys.executable,'-u','-B',str(R/('human_tdt_score_v2.py'if phase=='score'else 'human_tdt_train_v1.py'))],cwd=R,stdout=log,stderr=subprocess.STDOUT);put('HUMAN_TDT_'+phase.upper()+'_PID_V1.json',dict(pid=child.pid,start_ticks=ticks(child.pid)));code=child.wait()
  if code:break
put('HUMAN_TDT_DRIVER_TERMINAL_V1.json',dict(actual_exit_code=code,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
