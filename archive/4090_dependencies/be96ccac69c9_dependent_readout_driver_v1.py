import pathlib,json,hashlib,time,subprocess,os
r=pathlib.Path(__file__).resolve().parent
plan=json.loads((r/'DEPENDENT_READOUT_AUTOCONTINUE_PLAN_V1.json').read_text());h=plan['driver'];terminal=r/'FEATURE_TERMINAL_V1.json';missing=0
while not terminal.exists():
 if pathlib.Path(plan['stop_marker']).exists():raise SystemExit('User requested stop; readout notstarted')
 p=pathlib.Path('/proc')/str(h['pid'])/'stat'
 if p.exists():
  st=p.read_text().split()
  missing=missing+1 if st[21]!=h['start_ticks']or st[2]=='Z'else 0
 else:missing+=1
 if missing>2:raise RuntimeError('Expected feature driver disappeared without terminal; no restart or readout')
 time.sleep(10)
assert json.loads(terminal.read_text())['exitcode']==0,'Feature production failed; no readout'
assert(r/'LOCAL_DEPENDENCE_ALL_PUBLIC_FEATURES_SEALED_BEFORE_TARGET_FITS_V1.json').exists()
if pathlib.Path(plan['stop_marker']).exists():raise SystemExit('User requested stop; readout notstarted')
script=pathlib.Path(plan['readout']);assert hashlib.sha256(script.read_bytes()).hexdigest()==plan['readout_sha']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
with(r/'READOUT_WORKER_LOG_V1.txt').open('xb')as f:
 c=subprocess.Popen(['/home/lisiyi/miniconda3/envs/icassp/bin/python','-u','-B',str(script)],stdout=f,stderr=subprocess.STDOUT,env=env)
 with(r/'READOUT_CHILD_V1.json').open('x')as g:json.dump(dict(pid=c.pid,start_ticks=pathlib.Path('/proc/'+str(c.pid)+'/stat').read_text().split()[21]),g)
 ec=c.wait()
with(r/'READOUT_TERMINAL_V1.json').open('x')as f:json.dump(dict(exitcode=ec,time=time.time()),f)
