"""Frozen LMY encoder-cache extension: two VALID controls precede EVAL.

Uses the original public snapshot, precision, mono resampling, model forward,
and float16 cache format. Never imports an experiment training entry point.
"""
import argparse, ast, hashlib, json, os, sys, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for raw in iter(lambda:f.read(4*1024*1024),b''):h.update(raw)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--role',required=True);a=ap.parse_args()
    base=Path(a.base);cfg=json.loads((base/'protocol.json').read_text());rows=json.loads((base/'selection.json').read_text())
    out=base/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def save(name,value):
        t=out/(name+'.next');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(out/name)
    if cfg.get('vendor'):sys.path.insert(0,cfg['vendor'])
    import librosa,numpy as np,torch,transformers
    from transformers import AutoModelForCTC,AutoProcessor,ParakeetForTDT
    torch.set_num_threads(2)
    assert cfg['dtype']=='float16'
    device=torch.device('cuda');dtype=torch.float16
    torch.cuda.set_per_process_memory_fraction(cfg['memory_cap_MiB']*1024**2/torch.cuda.get_device_properties(0).total_memory)
    source=Path(cfg['source']);assert sha(source)==cfg['source_SHA']
    save('runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,librosa=librosa.__version__,source=str(source),source_SHA=sha(source)))
    processor=AutoProcessor.from_pretrained(cfg['model'],local_files_only=True)
    factory=AutoModelForCTC if cfg['family']=='Granite'else ParakeetForTDT
    model=factory.from_pretrained(cfg['model'],torch_dtype=dtype,local_files_only=True).to(device).eval().requires_grad_(False)
    if cfg['family']=='Granite':
        nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='prepare_inputs'];assert len(nodes)==1
        scope=dict(torch=torch,np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    def forward(path):
        wave,_=librosa.load(path,sr=16000,mono=True)
        if cfg['family']=='Granite':inputs=scope['prepare_inputs'](processor,wave,device,dtype)
        else:
            inputs=processor(audio=wave,sampling_rate=16000,return_tensors='pt')
            inputs={k:(v.to(device=device,dtype=dtype if k=='input_features'else v.dtype)if hasattr(v,'to')else v)for k,v in inputs.items()}
        with torch.inference_mode():
            if cfg['family']=='Granite':hidden=model(**inputs,output_hidden_states=True,return_dict=True).hidden_states[-1][0]
            else:
                result=model.get_audio_features(**inputs);hidden=result.last_hidden_state[0];mask=getattr(result,'attention_mask',None)
                if mask is not None:hidden=hidden[:int(mask[0].to(torch.int64).sum().item())]
        x=hidden.detach().to('cpu',dtype=torch.float16).numpy()
        assert x.ndim==2 and x.shape[1]==1024 and len(x)>0 and np.isfinite(x).all()
        return np.ascontiguousarray(x)
    ref=Path(cfg['validation_cache']);ix=json.loads((ref/'index.json').read_text());lookup={s:i for i,s in enumerate(ix['signal_ids'])}
    mm=np.memmap(ref/'features.f16',mode='r',dtype=np.float16,shape=(ix['total_tokens'],1024));proof=[]
    for row in rows[:2]:
        assert row['split']=='valid';j=lookup[row['signal_id']];offset=ix['offsets'][j];n=ix['lengths'][j]
        expected=np.asarray(mm[offset:offset+n]);x=forward(row['audio']);np.save(out/(row['signal_id']+'.npy'),x)
        gap=float(np.max(np.abs(x.astype(np.float64)-expected.astype(np.float64))))if x.shape==expected.shape else None
        proof.append(dict(signal=row['signal_id'],shape=list(x.shape),reference_shape=list(expected.shape),maxgap=gap,exact=bool(np.array_equal(x,expected))))
        save('VALID_control.json',dict(cases=proof))
        assert np.array_equal(x,expected),proof[-1]
    save('VALID_control_passed.json',dict(cases=proof,labels_read=False))
    done=[];lengths=[];offsets=[];total=0
    with(out/'features.f16.partial').open('xb')as f:
        for row in rows[2:]:
            assert row['split']=='eval';x=forward(row['audio']);f.write(x.tobytes());done.append(row['signal_id']);offsets.append(total);lengths.append(len(x));total+=len(x)
            if len(done)%8==0 or len(done)==2189:
                f.flush();save('progress.json',dict(done=len(done),total=2189,seconds=time.monotonic()-start))
        f.flush();os.fsync(f.fileno())
    assert len(done)==len(set(done))==2189
    partial=out/'features.f16.partial';assert partial.stat().st_size==total*1024*2;partial.replace(out/'features.f16')
    save('index.json',dict(model=cfg['model'],source_SHA=cfg['source_SHA'],sample_rate=16000,dtype='float16',dim=1024,split='eval',n=2189,total_tokens=total,signal_ids=done,offsets=offsets,lengths=lengths))
    save('complete.json',dict(rows=2189,training_updates=0,VALID_control=proof,seconds=time.monotonic()-start,outputs={name:sha(out/name)for name in ['index.json','features.f16']}))

if __name__=='__main__':main()
