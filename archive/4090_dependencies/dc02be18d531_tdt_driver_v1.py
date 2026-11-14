import os,sys,json,pathlib,time,subprocess,traceback
R=pathlib.Path(__file__).resolve().parent
def put(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2)
put('TDT_DRIVER_START_V1.json',dict(pid=os.getpid(),start_ticks=pathlib.Path('/proc/self/stat').read_text().split()[21]))
with(R/'tdt_worker.log').open('xb')as log:
 proc=subprocess.Popen([sys.executable,'-u','-B',str(R/'tdt_worker_v1.py')],stdout=log,stderr=subprocess.STDOUT,cwd=R)
 put('TDT_WORKER_START_V1.json',dict(pid=proc.pid,start_ticks=pathlib.Path('/proc/%d/stat'%proc.pid).read_text().split()[21]))
 code=proc.wait()
put('TDT_DRIVER_TERMINAL_V1.json',dict(actual_exit_code=code,formal_DEV_gold_read=False))
