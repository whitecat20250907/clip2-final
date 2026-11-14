from pathlib import Path
import os,json,time,argparse,hashlib
import numpy as np
def save(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--lane',type=int);ap.add_argument('--lanes',type=int);a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==p['uid'];manifest=json.loads((b/'audio_manifest.json').read_bytes());chunks={int(x.split('_')[1]) for x in p['roles']};rows=[x for i,x in enumerate(manifest['unique']) if i//64 in chunks][a.lane::a.lanes];start=time.monotonic();deadline=start+24*3600;pending=rows.copy();done=0;missing=0
import librosa
while pending and time.monotonic()<deadline:
 advanced=False
 for row in pending.copy():
  src=b/'vocals'/(row['audio_SHA']+'.npz')
  if not src.exists():continue
  ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
  if ram<48:time.sleep(10);break
  out=b/'pitch'/(row['audio_SHA']+'.npz');assert not out.exists()
  with np.load(src,allow_pickle=False) as z:wave=z['vocal'];sr=int(z['sr']);assert str(z['audio_SHA'])==row['audio_SHA'] and sr==16000
  began=time.monotonic();f0,voiced,prob=librosa.pyin(wave,fmin=65.406,fmax=2093.005,sr=sr,frame_length=2048,hop_length=320,center=True);assert len(f0)==len(voiced)==len(prob);times=np.arange(len(f0))*.02
  with out.with_suffix('.tmp').open('xb') as f:np.savez_compressed(f,f0=f0,voiced=voiced,voiced_probability=prob,times=times,audio_SHA=row['audio_SHA'],source_SHA=hashlib.sha256(src.read_bytes()).hexdigest())
  out.with_suffix('.tmp').replace(out);done+=1;missing+=int(not voiced.any());pending.remove(row);advanced=True
  save(b/'pitch_owners'/('lane'+str(a.lane)+'_progress.json'),dict(done=done,total=len(rows),pending=len(pending),all_unvoiced=missing,seconds=time.monotonic()-start,last_seconds=time.monotonic()-began,targets_read=False))
 if not advanced:time.sleep(15)
assert not pending,'24h deadline; retain partial outputs'
save(b/'pitch_owners'/('lane'+str(a.lane)+'_done.json'),dict(done=done,total=len(rows),seconds=time.monotonic()-start,all_unvoiced=missing,targets_read=False))
