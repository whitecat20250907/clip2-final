from pathlib import Path
import argparse,os,json,hashlib,time
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==p['uid'];rows=json.loads((b/'audio_manifest.json').read_bytes())['unique'];chunk=int(a.role.split('_')[1]);rows=rows[chunk*64:(chunk+1)*64];out=b/'tasks'/a.role;start=time.monotonic()
# This exact public checkpoint was SHA-verified and exercised in the pilot.
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import torch,soundfile as sf,soxr
from demucs.pretrained import get_model
from demucs.apply import apply_model
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.16)
cp=Path(p['Demucs']);assert sha(cp)==p['Demucs_SHA'];model=get_model('955717e8',repo=cp.parent).cuda().eval().requires_grad_(False);rate=int(model.samplerate);records=[]
for i,row in enumerate(rows):
 source=Path(row['audio']);assert sha(source)==row['audio_SHA'];wave,sr=sf.read(source,dtype='float32',always_2d=True);assert wave.shape[1]==2 and np.isfinite(wave).all();stereo=wave if sr==rate else soxr.resample(wave,sr,rate,quality='HQ').astype(np.float32);x=torch.from_numpy(stereo.T.copy()).cuda();ref=x.mean(0);mu,sd=ref.mean(),ref.std().clamp_min(1e-8)
 with torch.inference_mode():stems=apply_model(model,((x-mu)/sd)[None],shifts=0,split=True,overlap=.25,progress=False)[0]*sd+mu
 vi=list(model.sources).index('vocals');vocal=soxr.resample(stems[vi].mean(0).cpu().numpy(),rate,16000,quality='HQ').astype(np.float32);assert np.isfinite(vocal).all() and abs(len(vocal)/16000-len(wave)/sr)<.003
 dest=b/'vocals'/(row['audio_SHA']+'.npz')
 with dest.with_suffix('.tmp').open('xb') as f:np.savez_compressed(f,vocal=vocal,sr=16000,audio_SHA=row['audio_SHA'],Demucs_SHA=p['Demucs_SHA'])
 dest.with_suffix('.tmp').replace(dest);records.append(dict(audio_SHA=row['audio_SHA'],output_SHA=sha(dest),samples=len(vocal)));del stems,x
 save(out/'progress.json',dict(rows=i+1,total=len(rows),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/1024**2))
save(out/'result.json',dict(rows=len(rows),seconds=time.monotonic()-start,records=records,worker_SHA=sha(__file__),Demucs_SHA=p['Demucs_SHA'],peak_MiB=torch.cuda.max_memory_allocated()/1024**2,targets_read=False))
