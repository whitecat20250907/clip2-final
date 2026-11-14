"""Frozen Y4: original Demucs waveform, independent pYIN, original RMVPE readout."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_text());out=b/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
 def sha(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
 def save(name,v):
  q=out/(name+'.next');q.write_text(json.dumps(v,indent=2));q.replace(out/name)
 sys.path[:0]=p['runtime'].split(':');os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
 import numpy as np,torch,soundfile as sf,soxr,librosa
 from demucs.pretrained import get_model
 from demucs.apply import apply_model
 from concurrent.futures import ThreadPoolExecutor
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.16)
 rows=json.loads((b/'selection.json').read_text());expected={r['signal_id']:r for r in json.loads((b/'VALID_expected.json').read_text())};assert len(rows)==2191 and set(expected)=={r['signal_id']for r in rows[:2]}
 cp=Path(p['Demucs']);assert sha(cp)==p['Demucs_SHA'];model=get_model('955717e8',repo=cp.parent).cuda().eval().requires_grad_(False);rate=int(model.samplerate)
 for kind in ['vocals','pitch']:(b/'canonical'/kind).mkdir(parents=True,exist_ok=True)
 proof=[];seen=set();pending=[];finished=0
 def pitch(row):
  key=row['audio_SHA'];src=b/'canonical/vocals'/(key+'.npz');dest=b/'canonical/pitch'/(key+'.npz')
  with np.load(src,allow_pickle=False)as z:wave=z['vocal'];sr=int(z['sr']);assert str(z['audio_SHA'])==key and sr==16000
  f0,voiced,prob=librosa.pyin(wave,fmin=65.406,fmax=2093.005,sr=sr,frame_length=2048,hop_length=320,center=True);assert len(f0)==len(voiced)==len(prob);times=np.arange(len(f0))*.02
  with dest.open('xb')as f:np.savez_compressed(f,f0=f0,voiced=voiced,voiced_probability=prob,times=times,audio_SHA=key,source_SHA=sha(src))
  if row['split']=='valid':
   with np.load(expected[row['signal_id']]['source_pitch'],allow_pickle=False)as old:
    assert np.array_equal(voiced,old['voiced'])and np.array_equal(times,old['times'])
    assert np.allclose(f0,old['f0'],rtol=0,atol=1e-6,equal_nan=True)and np.allclose(prob,old['voiced_probability'],rtol=0,atol=1e-6,equal_nan=True)
  return key
 with ThreadPoolExecutor(max_workers=2)as executor:
  for i,row in enumerate(rows):
   if i==2:assert len(proof)==2;save('VALID_stem_pitch_control_passed.json',dict(cases=proof,training_updates=0))
   key=row['audio_SHA']
   if key in seen:continue
   source=Path(row['audio']);assert sha(source)==key;wave,sr=sf.read(source,dtype='float32',always_2d=True);assert wave.shape[1]==2 and np.isfinite(wave).all();stereo=wave if sr==rate else soxr.resample(wave,sr,rate,quality='HQ').astype(np.float32);x=torch.from_numpy(stereo.T.copy()).cuda();ref=x.mean(0);mu,sd=ref.mean(),ref.std().clamp_min(1e-8)
   with torch.inference_mode():stems=apply_model(model,((x-mu)/sd)[None],shifts=0,split=True,overlap=.25,progress=False)[0]*sd+mu
   vi=list(model.sources).index('vocals');vocal=soxr.resample(stems[vi].mean(0).cpu().numpy(),rate,16000,quality='HQ').astype(np.float32);assert np.isfinite(vocal).all()and abs(len(vocal)/16000-len(wave)/sr)<.003
   dest=b/'canonical/vocals'/(key+'.npz')
   with dest.open('xb')as f:np.savez_compressed(f,vocal=vocal,sr=16000,audio_SHA=key,Demucs_SHA=p['Demucs_SHA'])
   if row['split']=='valid':
    with np.load(expected[row['signal_id']]['source_vocal'],allow_pickle=False)as z:old=z['vocal'];assert old.shape==vocal.shape;gap=float(np.max(abs(old-vocal)));assert gap<=1e-6
    pitch(row);proof.append(dict(signal_id=row['signal_id'],wave_maxgap=gap,pitch_match=True))
   else:pending.append(executor.submit(pitch,row))
   seen.add(key);del stems,x
   if len(pending)>=8:pending.pop(0).result();finished+=1
   save('stem_pitch_progress.json',dict(unique_stems=len(seen),unique_total=len({r['audio_SHA']for r in rows}),pitch_finished=finished,seconds=time.monotonic()-began))
  for f in pending:f.result();finished+=1
 del model;torch.cuda.empty_cache()
 save('stems_pitch_complete.json',dict(unique_audio=len(seen),seconds=time.monotonic()-began))
 source=b/'original_y4_worker.py';assert sha(source)==p['original_worker_SHA'];text=source.read_text();checks=[]
 def validate(row,values):
  if row['split']=='eval':assert len(checks)==2;return
  xx=np.asarray(values);old=np.asarray(expected[row['signal_id']]['x']);assert xx.shape==old.shape;gap=float(abs(xx-old).max());checks.append(dict(signal_id=row['signal_id'],maxgap=gap));save('VALID_Y4_control.json',checks);assert gap<=1e-6
  if len(checks)==2:save('VALID_Y4_control_passed.json',dict(cases=checks,training_updates=0))
 patches=[("if i%10==0:save(","validate(r,values)\n        if i%10==0:save("),("(out/'evidence.jsonl').open('w')","(out/'evidence.jsonl').open('x')")]
 for old,new in patches:assert text.count(old)==1;text=text.replace(old,new)
 save('engineering_patch.json',dict(original_SHA=p['original_worker_SHA'],patches=patches,model_math_changed=False))
 exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source),validate=validate))
 assert len(checks)==2;complete=json.loads((out/'complete.json').read_text());complete.update(EVAL_rows=2189,VALID_controls=2,training_updates=0,total_seconds=time.monotonic()-began);save('complete.json',complete)

if __name__=='__main__':main()
