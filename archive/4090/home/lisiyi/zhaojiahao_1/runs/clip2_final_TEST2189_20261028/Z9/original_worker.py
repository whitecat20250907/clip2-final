"""PANNs public Cnn14 qualification and windowed Audio2 scene features."""
from pathlib import Path
import os,sys,json,argparse,time,subprocess,hashlib,csv,io
def save(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x));q.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==p['owner_uid'];out=b/'tasks'/a.role;out.mkdir(exist_ok=True,parents=True);assets=b/'scene_assets';assets.mkdir(exist_ok=True)
    if a.role=='scene_assets':
        import requests
        jobs=[('models.py','https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/pytorch/models.py'),('pytorch_utils.py','https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/pytorch/pytorch_utils.py'),('classes.csv','https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv'),('Cnn14_mAP0.431.pth','https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1')]
        for name,url in jobs:
            q=assets/name
            if q.exists():continue
            with requests.get(url,stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                with q.with_suffix('.part').open('wb') as f:
                    for chunk in r.iter_content(1024**2):f.write(chunk)
                q.with_suffix('.part').replace(q)
        runtime=assets/'runtime';runtime.mkdir(exist_ok=True)
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--target',str(runtime),'torchlibrosa==0.1.0'],timeout=240,check=True)
        save(out/'complete.json',dict(files={n:dict(SHA=hashlib.sha256((assets/n).read_bytes()).hexdigest(),bytes=(assets/n).stat().st_size,url=u) for n,u in jobs}));(b/'scene_ready').write_text('ready');return
    sys.path[:0]=[str(assets/'runtime'),str(assets)];import numpy as np,torch,soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    from models import Cnn14
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);model=Cnn14(sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000,classes_num=527)
    checkpoint=torch.load(assets/'Cnn14_mAP0.431.pth',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model']);model=model.cuda().eval().requires_grad_(False)
    classes=list(csv.DictReader((assets/'classes.csv').open()));names=[r['display_name'] for r in classes]
    wanted=['Chatter','Hubbub, speech noise, speech babble','Cutlery, silverware','Dishes, pots, and pans','Singing'];indices=[names.index(n) for n in wanted]
    rows=json.loads((b/'scene_selection.json').read_bytes());k=int(a.role.rsplit('_',1)[1]);rows=rows[k::p['scene_parts']];start=time.monotonic();data=Path(p['data_root'])
    with (out/'evidence.jsonl').open('w') as f:
        for i,r in enumerate(rows):
            sid=r['signal_id'];path=data/r['split']/'unprocessed'/(sid+'_unproc.flac')
            if not path.exists():
                candidates=[data/r['split'].lower()/'audio'/('Audio2_'+sid+'.wav'),data/r['split'].lower()/'audio2'/(sid+'.wav')]
                path=next((q for q in candidates if q.exists()),path)
            # Exact data naming supplied by prior audited selection when available.
            if r.get('audio2_relative'):path=data/r['audio2_relative']
            wave,sr=sf.read(path,dtype='float32',always_2d=True);ear=[]
            for e in range(wave.shape[1]):
                w=wave[:,e];gg=gcd(int(sr),32000);w=resample_poly(w,32000//gg,int(sr)//gg);windows=[w[j:j+64000] for j in range(0,len(w),64000)];x=np.stack([np.pad(v,(0,max(0,64000-len(v))))[:64000] for v in windows])
                with torch.inference_mode():q=model(torch.from_numpy(x).cuda())['clipwise_output'][:,indices].cpu().numpy()
                ear.append(np.median(q,axis=0))
            q=np.mean(ear,axis=0);x=np.r_[q[:4],q[0]/max(q[0]+q[2]+q[3],1e-5),q[1]/max(q[1]+q[2]+q[3],1e-5)]
            f.write(json.dumps(dict(audio_key=r['audio_key'],signal_id=sid,x=x.tolist(),singing=float(q[4]),ear_difference=float(np.max(abs(ear[0]-ear[-1]))),seconds=len(wave)/sr,window_seconds=2,clipwise_window_aggregation='median, final short window zero-padded'))+'\n');f.flush()
            if i%5==0:save(out/'progress.json',dict(done=i+1,total=len(rows),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/1024**2))
    save(out/'complete.json',dict(rows=len(rows),seconds=time.monotonic()-start,classes=wanted,peak_MiB=torch.cuda.max_memory_allocated()/1024**2,model='Cnn14 clipwise 0.431, NOT framewise model',limitations='Background class posterior is not known target-language masking; singing is only an activity proxy'))
if __name__=='__main__':main()
