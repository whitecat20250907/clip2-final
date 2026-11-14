import os,sys,pathlib,json,subprocess,time
R=pathlib.Path(__file__).resolve().parent
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True)
for phase,worker in [('train','projection_train_v1.py'),('score','projection_score_v1.py')]:
 with(R/('PROJ_TDT_'+phase.upper()+'_LOG_V1.txt')).open('xb')as log:
  c=subprocess.Popen([sys.executable,'-u','-B',str(R/worker)],cwd=R,stdout=log,stderr=subprocess.STDOUT);put('PROJ_TDT_'+phase.upper()+'_PID_V1.json',dict(pid=c.pid,start_ticks=pathlib.Path('/proc/%d/stat'%c.pid).read_text().split()[21]));code=c.wait()
 if code:break
put('PROJ_TDT_DRIVER_TERMINAL_V1.json',dict(actual_exit_code=code,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
