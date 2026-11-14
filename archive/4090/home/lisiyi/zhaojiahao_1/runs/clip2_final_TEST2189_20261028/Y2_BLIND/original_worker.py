from pathlib import Path
import os,sys,argparse,json,time
ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==p['owner_uid']
import numpy as np,torch,soundfile as sf,soxr
from v10_Y2_blind_math_20261022 import features,golden
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.04);out=b/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);rows=json.loads((b/'selection.json').read_bytes())[int(a.role.split('_')[-1])::p['parts']];start=time.monotonic()
def save(name,data):
    q=out/(name+'.tmp');q.write_text(json.dumps(data));q.replace(out/name)
save('qualification.json',golden())
with (out/'evidence.jsonl').open('w') as f:
    for i,r in enumerate(rows):
        path=Path(p['data_root'])/r['split']/'signals'/(r['signal_id']+'.flac');wave,sr=sf.read(path,dtype='float32',always_2d=True);assert wave.shape[1]==2
        if sr!=16000:wave=soxr.resample(wave,sr,16000,quality='HQ')
        x,t=features(wave);wp=[]
        for weights in r['word_profile']:
            weights=np.asarray(weights,float);weights/=max(weights.sum(),1e-12);centers=(np.arange(len(weights))+.5)*.08;grid=np.interp(t,centers,weights,left=0,right=0);mass=grid.sum()
            wp.append((grid[:,None]*x).sum(0).tolist() if mass<=0 else ((grid[:,None]*x).sum(0)/mass).tolist())
        f.write(json.dumps(dict(signal_id=r['signal_id'],x=wp,frames=len(t),reference_audio_read=False))+'\n');f.flush()
        if i%10==0:save('progress.json',dict(done=i+1,total=len(rows),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/1024**2))
save('complete.json',dict(rows=len(rows),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/1024**2,models_trained=False,Audio2_read=False))
