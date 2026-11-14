"""Compare old-runtime decoded/resampled waves with pinned reference1.0 waves."""
import os,sys,json,time,traceback,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;REF=H/'waveform16_reference_v1';OUT=H/'waveform16_comparison_v1';OUT.mkdir(exist_ok=False)
for key in ['NUMBA_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','MPLCONFIGDIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
sys.dont_write_bytecode=True
def save(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
try:
 import numpy as np,librosa,soundfile as sf,soxr
 for j in range(180):
  if (REF/'exitcode').exists():break
  try:os.kill(2904105,0)
  except ProcessLookupError:raise RuntimeError('Reference waveform worker missing without exit marker')
  time.sleep(10)
 assert (REF/'exitcode').read_text().strip()=='0'
 report=json.loads((REF/'report.json').read_text())
 p=REF/'reference_waves16.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==report['waves_npz_sha256']
 with np.load(p,allow_pickle=False) as z:
  ids=z['signals'].tolist();ll=z['lengths'].astype(np.int64);vv=z['waves'].copy()
 offsets=np.r_[0,np.cumsum(ll[:-1])];records=[];data=Path('/home/lisiyi/ICASSP2_2026/data/clip2/train/signals')
 for j,sid in enumerate(ids):
  path=data/(sid+'.flac');wav,_=librosa.load(path,sr=16000,mono=True);ref=vv[offsets[j]:offsets[j]+ll[j]];assert wav.shape==ref.shape
  raw,sr0=sf.read(path,dtype='float32',always_2d=True);r=report['records'][j];assert r['signal']==sid
  delta=wav.astype(np.float64)-ref.astype(np.float64)
  records.append(dict(signal=sid,decoded_stereo_float32_hash_equal=hashlib.sha256(raw.tobytes()).hexdigest()==r['decoded_stereo_float32_sha256'],max_abs_wave_gap=float(abs(delta).max()),relative_wave_RMSE=float(np.linalg.norm(delta)/max(np.linalg.norm(ref.astype(np.float64)),1e-30)),wave_values_equal=bool(np.array_equal(wav,ref))))
 save('report.json',dict(status='waveform16_old_vs_reference_comparison_complete',records=records,old_runtime=dict(librosa=librosa.__version__,soxr=soxr.__version__,soundfile=sf.__version__,numpy=np.__version__),reference_runtime={k:report[k] for k in ['librosa','soxr','soundfile','numpy']},labels_read=False,model_loaded=False))
 (OUT/'exitcode').write_text('0');print('WAVE16_COMPARISON_COMPLETE',json.dumps(records),flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
