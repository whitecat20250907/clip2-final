"""Reference 1.0 audio load versus installed0.11, no labels or model scores."""
import os,sys,json,time,hashlib,traceback,datetime
from pathlib import Path
H=Path(__file__).resolve().parent;OUT=H/'waveform16_reference_v1';OUT.mkdir(exist_ok=False)
for key in ['NUMBA_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','MPLCONFIGDIR']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
sys.dont_write_bytecode=True;sys.path.insert(0,str(H))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
try:
 import numpy as np,librosa,soundfile as sf,soxr
 import audio_load_reference100_v1 as ref
 ids=json.loads((H/'numeric_audio16_v1/protocol.json').read_text())['ids'];assert len(ids)==16
 waves=[];lens=[];records=[]
 data=Path('/home/lisiyi/ICASSP2_2026/data/clip2/train/signals')
 for sid in ids:
  p=data/(sid+'.flac');wav,sr=ref.load(p,sr=16000,mono=True);old,_=librosa.load(p,sr=16000,mono=True);raw,sr0=sf.read(p,dtype='float32',always_2d=True)
  assert sr==16000 and wav.shape==old.shape
  gap=float(abs(wav-old).max());assert gap==0
  waves.append(wav);lens.append(len(wav));records.append(dict(signal=sid,audio_sha256=sha(p),native_sr=sr0,native_channels=raw.shape[1],decoded_stereo_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),wave_sha256=hashlib.sha256(wav.tobytes()).hexdigest(),reference100_vs_installed011_gap=gap))
 with (OUT/'reference_waves16.npz').open('xb') as f:np.savez_compressed(f,signals=np.array(ids),lengths=np.array(lens,np.int64),waves=np.concatenate(waves))
 save('report.json',dict(status='reference100_waves16_ready',records=records,librosa=librosa.__version__,soxr=soxr.__version__,soundfile=sf.__version__,numpy=np.__version__,source_module_sha256=sha(H/'audio_load_reference100_v1.py'),waves_npz_sha256=sha(OUT/'reference_waves16.npz'),no_training=True,no_labels=True))
 (OUT/'exitcode').write_text('0');print('REFERENCE_WAVE16_COMPLETE',flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
