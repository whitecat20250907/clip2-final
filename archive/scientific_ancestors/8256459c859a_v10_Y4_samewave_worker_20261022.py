"""Canonical cached stems only; comparison never interpolates unvoiced zeros."""
from pathlib import Path
import os,sys,argparse,json,time,hashlib
ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==p['owner_uid'];sys.path[:0]=p['runtime'].split(':')
import numpy as np,torch
from rmvpe_vendor import RMVPE
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);model=RMVPE(p['model'],False,device='cuda');out=b/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic();rows=json.loads((b/'selection.json').read_bytes())
if not a.role.startswith('qual'):rows=[r for r in rows if int(r['audio_SHA'][:12],16)%p['parts']==int(a.role.rsplit('_',1)[1])]
def save(name,value):
    tmp=out/(name+'.tmp');tmp.write_text(json.dumps(value));tmp.replace(out/name)
def source(key,kind):
    for root in p['canonical_roots']:
        path=Path(root)/kind/(key+'.npz')
        if path.exists():return path
    raise FileNotFoundError('No canonical '+kind+' for '+key)
def nearest_pitch(f0,times,target):
    # Choose a real independently estimated sample, including its voicing.
    ix=np.searchsorted(times,target);ix=np.clip(ix,0,len(times)-1);left=np.clip(ix-1,0,len(times)-1);ix=np.where(abs(times[left]-target)<=abs(times[ix]-target),left,ix);v=f0[ix].copy();v[(target<times[0])|(target>times[-1])]=np.nan;return v
cache={};comparisons=[];frame=out/'frames';frame.mkdir(exist_ok=True)
with (out/'evidence.jsonl').open('w') as f:
    for i,r in enumerate(rows):
        key=r['audio_SHA'];vocal=source(key,'vocals');pitch=source(key,'pitch')
        if key not in cache:
            vocal_SHA=hashlib.sha256(vocal.read_bytes()).hexdigest()
            with np.load(vocal,allow_pickle=False) as z:wave=z['vocal'].astype('float32');sr=int(z['sr']);assert str(z['audio_SHA'])==key and sr==16000
            with np.load(pitch,allow_pickle=False) as z:assert str(z['source_SHA'])==vocal_SHA;pf=z['f0'].copy();pt=z['times'].copy()
            with torch.inference_mode():f0=model.infer_from_audio(wave,thred=.03)
            times=np.arange(len(f0))*.01;other=nearest_pitch(pf,pt,times);both=(f0>0)&np.isfinite(other)&(other>0);distance=np.full(len(f0),np.nan);distance[both]=abs(1200*np.log2(f0[both]/other[both]));agree=both&(distance<=600);legacy=np.interp(times,pt,np.nan_to_num(pf,nan=0),left=0,right=0);oldboth=(f0>0)&(legacy>0)
            comparison=dict(both_voiced_frames=int(both.sum()),RMVPE_voiced_frames=int((f0>0).sum()),octave_disagreement_frames=int(np.sum(distance[both]>600)),median_cents=float(np.median(distance[both])) if both.any() else None,legacy_zero_interpolated_both_frames=int(oldboth.sum()),legacy_zero_interpolated_disagreement_frames=int(np.sum(abs(1200*np.log2(f0[oldboth]/legacy[oldboth]))>600)),same_waveform_SHA_verified=True)
            np.savez_compressed(frame/(key+'.npz'),f0=f0,times=times,independent_f0=other,agreed=agree,audio_SHA=key,vocal_SHA=vocal_SHA)
            cache[key]=(f0,times,other,agree,comparison);comparisons.append(comparison)
        f0,times,other,agree,comparison=cache[key];values=[];control=[]
        for lo,hi in r['intervals']:
            def summarize(start,end):
                use=(times>=start)&(times<=end);good=f0[use&agree];active=use&(f0>0);return [float(np.median(good)/1000) if len(good) else 0.,float(np.quantile(good,.9)/1000) if len(good) else 0.,float(np.mean(good>550)) if len(good) else 0.,float(np.mean(good>650)) if len(good) else 0.,float(active.sum()/max(use.sum(),1)),float(np.sum(use&agree)/max(active.sum(),1))]
            values.append(summarize(lo,hi));shift=(lo+times[-1]/3)%max(times[-1],.01);control.append(summarize(shift,min(shift+hi-lo,times[-1])))
        f.write(json.dumps(dict(signal_id=r['signal_id'],audio_SHA=key,x=values,time_shift_control=control,tracker_comparison=comparison,source_vocal=str(vocal),source_pitch=str(pitch),raw_mixture_fallback=False))+'\n');f.flush()
        if i%10==0:save('progress.json',dict(done=i+1,total=len(rows),unique_audio=len(cache),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/1024**2))
save('complete.json',dict(rows=len(rows),unique_audio=len(cache),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/1024**2,comparisons=comparisons,targets_read=False,all_canonical_stems=True))
