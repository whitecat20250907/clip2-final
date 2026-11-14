"""Frozen Whisper response to a new blind EC front end; no A2 use."""
from pathlib import Path
import os,sys,json,argparse,time
ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assert b.stat().st_uid==os.getuid()==p['owner_uid'];sys.path.insert(0,p['root']+'/03_my_improvements')
import numpy as np,torch,soundfile as sf,soxr
from scipy.signal import correlate
from transformers import WhisperForConditionalGeneration,WhisperProcessor,AutoTokenizer
from d6e_word_v1.listener_pilot import tokenize
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25);out=b/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic();rows=json.loads((b/'selection.json').read_bytes())[int(a.role.rsplit('_',1)[1])::p['parts']]
proc=WhisperProcessor.from_pretrained(p['model'],local_files_only=True);proc.tokenizer=AutoTokenizer.from_pretrained(p['model'],local_files_only=True,use_fast=True);proc.tokenizer.set_prefix_tokens(language='english',task='transcribe',predict_timestamps=False);model=WhisperForConditionalGeneration.from_pretrained(p['model'],local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
def front(wave):
    result=wave.copy();activity=[]
    for lo in range(0,len(wave),8000):
        hi=min(lo+8000,len(wave));x=wave[lo:hi].astype(float)
        if len(x)<64:continue
        energy=np.sum(x*x,0);cross=correlate(x[:,0],x[:,1],mode='full',method='fft');center=len(x)-1;lag=int(np.argmax(cross[center-11:center+12]))-11;r=np.interp(np.arange(len(x))-lag,np.arange(len(x)),x[:,1],left=0,right=0);cov=float(x[:,0]@r);coh=cov/max(np.sqrt(energy[0]*(r@r)),1e-12)
        gain=float(np.clip(cov/max(r@r,1e-12),.316,3.162));res=x[:,0]-gain*r;ratio=float((res@res)/max(energy[0],1e-12));active=.15<=ratio<=.8 and abs(coh)<.995
        if active:
            y=res*np.clip(np.sqrt(energy[0]/max(res@res,1e-12)),.5,2);rr=np.interp(np.arange(len(x))+lag,np.arange(len(x)),x[:,0],left=0,right=0);g2=float(np.clip(x[:,1]@rr/max(rr@rr,1e-12),.316,3.162));right=x[:,1]-g2*rr;right*=np.clip(np.sqrt(energy[1]/max(right@right,1e-12)),.5,2);window=np.ones(len(x));edge=min(160,len(x)//4);window[:edge]=np.linspace(0,1,edge);window[-edge:]=window[:edge][::-1];result[lo:hi,0]=x[:,0]+.5*window*(y-x[:,0]);result[lo:hi,1]=x[:,1]+.5*window*(right-x[:,1])
        activity.append(active)
    assert np.isfinite(result).all();return result,float(np.mean(activity)) if activity else 0.
test=np.random.default_rng(22).normal(0,.1,(32000,1)).repeat(2,1).astype(np.float32);same,activity=front(test);assert activity==0 and np.array_equal(test,same)
def score(wave,text):
    decoder,target,_,mapping=tokenize(proc.tokenizer,text,'cuda',offsets=True);n=len(text.split());fe=proc.feature_extractor(wave,sampling_rate=16000,return_attention_mask=True,return_tensors='pt')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logits=model(input_features=fe.input_features.cuda(),attention_mask=fe.attention_mask.cuda(),decoder_input_ids=decoder,use_cache=False).logits
    lp=logits.float().log_softmax(-1).gather(-1,target.unsqueeze(-1))[0,:,0].cpu().numpy();return np.asarray([lp[np.asarray(mapping)==j].mean() for j in range(n)])
def save(name,v):
    tmp=out/(name+'.tmp');tmp.write_text(json.dumps(v));tmp.replace(out/name)
with (out/'evidence.jsonl').open('w') as f:
    for i,r in enumerate(rows):
        wave,sr=sf.read(Path(p['data_root'])/r['split']/'signals'/(r['signal_id']+'.flac'),dtype='float32',always_2d=True)
        if sr!=16000:wave=soxr.resample(wave,sr,16000,quality='HQ')
        enhanced,activity=front(wave);raw=np.stack([score(wave[:,ear],r['prompt']) for ear in [0,1]],1);new=np.stack([score(enhanced[:,ear],r['prompt']) for ear in [0,1]],1);delta=np.clip(new-raw,-10,10)/10;feat=np.column_stack([delta,abs(delta[:,0]-delta[:,1]),np.full(len(delta),activity)]);assert np.isfinite(feat).all()
        f.write(json.dumps(dict(signal_id=r['signal_id'],x=feat.tolist(),active_fraction=activity))+'\n');f.flush()
        if i%5==0:save('progress.json',dict(done=i+1,total=len(rows),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/1024**2))
save('complete.json',dict(rows=len(rows),seconds=time.monotonic()-start,duplicate_ear_exact_fallback=True,peak_MiB=torch.cuda.max_memory_allocated()/1024**2,Audio2_read=False,listener_frozen=True))
