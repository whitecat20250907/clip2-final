"""Frozen latest-LMY Qwen39 and physical-stereo78 public feature export.

Original numerical functions are extracted from pinned sources without running
their peer-directory initialization or training routines. Two existing VALID
feature rows must pass before any EVAL inference. No labels or fitting.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import sys
import time
import types


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True)
    ap.add_argument('--role',required=True)
    args=ap.parse_args()
    base=pathlib.Path(args.base)
    cfg=json.loads((base/'protocol.json').read_text())
    out=base/'tasks'/args.role
    out.mkdir(parents=True,exist_ok=True)
    for name in ['rows','nulls']:(out/name).mkdir(exist_ok=True)
    start=time.monotonic()
    def save(name,obj):
        temp=out/(name+'.next')
        temp.write_text(json.dumps(obj,indent=2,allow_nan=False))
        temp.replace(out/name)
    def note(stage,**kwargs):
        save('progress.json',dict(stage=stage,seconds=time.monotonic()-start,**kwargs))
    note('imports')
    sys.path.insert(0,cfg['runtime_site'])
    import numpy as np
    import torch
    import soundfile as sf
    import soxr
    import transformers
    from torch import nn
    from torch.nn import functional as F
    from safetensors import safe_open
    import transformers.utils.import_utils as availability
    availability._torchvision_available=False
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    free,total=torch.cuda.mem_get_info()
    assert free>=5*1024**3
    torch.cuda.set_per_process_memory_fraction(4*1024**3/total)
    env=dict(np=np,torch=torch,sf=sf,soxr=soxr,transformers=transformers,
             nn=nn,F=F,safe_open=safe_open,os=os,sys=sys,pathlib=pathlib,
             json=json,hashlib=hashlib,math=math,types=types,importlib=__import__('importlib'),
             Q=pathlib.Path(cfg['runtime_root']),MODEL=pathlib.Path(cfg['model']))
    selected={
      'common_v1.py':['sha','read','mod','imports','own_mask','mask_hook','load_tower',
                       'prepare','forward','LoRA','insert_lora','ctc_logp','ctc'],
      'prefix_common_v1.py':['PrefixCut','Prefix','tail','mathctx'],
      'score_math_v1.py':['greedy','edits','minimum'],
      'study_common_v1.py':['public_hash','tensor_hash','set_delta','frame_forward',
                            'lp_forward','features','disagreement','dual_features']}
    for name,names in selected.items():
        path=pathlib.Path(cfg['sources'][name]['path'])
        raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==cfg['sources'][name]['SHA']
        nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
        assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    records=json.loads((base/'selection.json').read_text())
    assert len(records)==2191 and len({r['signal']for r in records})==2191
    assert [r['role']for r in records[:2]]==['VALID_CONTROL']*2
    assert all(r['role']=='EVAL' for r in records[2:])
    assert not any({'correctness','response','target','labels'}&r.keys()for r in records)
    phones={p:i+1 for i,p in enumerate('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())}
    dictionary={}
    for line in pathlib.Path(cfg['dictionary']).read_text().splitlines():
        fields=line.split()
        if len(fields)<2 or fields[0].startswith(';;;'):continue
        word=re.sub(r'\(\d+\)$','',fields[0]).lower()
        seq=[re.sub(r'\d','',p)for p in fields[1:]]
        if word not in dictionary and all(p in phones for p in seq):dictionary[word]=[phones[p]for p in seq]
    note('loading_same_frozen_Qwen')
    tower,extractor,_,_=env['load_tower']()
    tower.to('cuda:0');tower.config._attn_implementation='sdpa'
    before=env['public_hash'](tower)
    mods=env['insert_lora'](tower)
    prefix=env['Prefix'](tower)
    checkpoints={};heads={};expected={}
    for seed in [44,45]:
        c=cfg['seeds'][str(seed)]
        ck=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['seed']==seed and ck['steps']==945 and ck['arm']=='dual_reference_human'
        checkpoints[seed]=ck
        hs=nn.ModuleDict({n:nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,64),nn.GELU(),nn.Linear(64,40))for n in ['ref','response']})
        hs.load_state_dict(ck['heads'],strict=True)
        heads[seed]=hs.to('cuda:0').eval().requires_grad_(False)
        expected[seed]={}
        for name in ['q39','stereo78']:
            with np.load(c['validation_'+name],allow_pickle=False)as z:
                ids=z['signals'].tolist()
                expected[seed][name]={r['signal']:z['features'][ids.index(r['signal'])].copy()for r in records[:2]}
    tower.eval().requires_grad_(False)
    nulls={};proof=[];results={s:[]for s in [44,45]}
    def prep(wave,sr):
        wave=np.ascontiguousarray(wave,np.float32)
        if sr!=16000:wave=soxr.resample(wave,sr,16000,quality='HQ').astype(np.float32)
        wave=np.pad(wave,(0,(-len(wave))%160))
        z=extractor(wave,sampling_rate=16000,padding='longest',truncation=False,return_attention_mask=True,return_tensors='pt')
        mel=z['input_features'][0];nf=int(z['attention_mask'][0].sum())
        assert mel.shape==(128,nf)and nf==len(wave)//160
        return mel,nf
    with torch.no_grad(),env['mathctx']():
        for index,row in enumerate(records):
            if index==2:
                assert len(proof)==4
                save('VALID_control_passed.json',dict(cases=proof,training_updates=0,labels_read=False))
            words=row['prompt'].split()
            assert len(words)==row['n_words']
            ref=None if any(w not in dictionary for w in words)else[v for w in words for v in dictionary[w]]
            public=dict(n_words=len(words),severity=row['severity'],reference=ref)
            audio=pathlib.Path(row['audio'])
            audio_sha=env['sha'](audio)
            wave,sr=sf.read(str(audio),dtype='float32',always_2d=True)
            assert wave.shape[1]==2 and np.isfinite(wave).all()
            data=[];nfs=[]
            for w in [wave.mean(1,dtype=np.float32),wave[:,0],wave[:,1]]:
                mel,nf=prep(w,sr)
                h,cu=prefix(mel.to('cuda:0'))
                data.append(dict(h=h.numpy(),cu=cu.numpy(),mel=nf));nfs.append(nf)
            assert nfs[0]==nfs[1]==nfs[2]
            nf=nfs[0]
            if nf not in nulls:
                z=extractor(np.zeros(nf*160,np.float32),sampling_rate=16000,padding='longest',truncation=False,return_attention_mask=True,return_tensors='pt')
                mel=z['input_features'][0];assert mel.shape==(128,nf)
                h,cu=prefix(mel.to('cuda:0'));null_item=dict(h=h.numpy(),cu=cu.numpy(),mel=nf)
                nulls[nf]={}
                for seed in [44,45]:
                    env['set_delta'](mods,checkpoints[seed]['audio_delta'])
                    nulls[nf][seed]=np.stack([x.numpy().copy()for x in env['lp_forward'](tower,heads[seed],null_item)],axis=1)
                np.savez_compressed(out/'nulls'/(str(nf)+'.npz'),mel=nf,lp44=nulls[nf][44],lp45=nulls[nf][45])
            values={}
            for seed in [44,45]:
                env['set_delta'](mods,checkpoints[seed]['audio_delta'])
                lp=[np.stack([x.numpy().copy()for x in env['lp_forward'](tower,heads[seed],item)],axis=1)for item in data]
                mono=env['dual_features'](lp[0],nulls[nf][seed],public)
                stereo=env['dual_features'](np.logaddexp(lp[1],lp[2])-np.log(2.),nulls[nf][seed],public)
                value=np.r_[mono,stereo];assert value.shape==(78,)
                values['features'+str(seed)]=value
                if index<2:
                    target=expected[seed]['stereo78'][row['signal']]
                    assert np.array_equal(np.isnan(value),np.isnan(target))
                    gap=float(np.nanmax(abs(value-target)))
                    mono_gap=float(np.nanmax(abs(mono-expected[seed]['q39'][row['signal']])))
                    assert gap<=1e-4 and mono_gap<=1e-6,(seed,row['signal'],gap,mono_gap)
                    proof.append(dict(seed=seed,signal=row['signal'],q39_maxgap=mono_gap,stereo78_maxgap=gap))
                else:results[seed].append(value)
            np.savez_compressed(out/'rows'/(row['signal']+'.npz'),signal=row['signal'],audio_SHA=audio_sha,mel=nf,**values)
            note('VALID_control'if index<2 else'EVAL_inference',done=max(0,index-1),total=2189,signal=row['signal'])
    prefix.hook.remove()
    assert env['public_hash'](tower)==before
    for seed in [44,45]:
        assert all(torch.equal(v.cpu(),checkpoints[seed]['heads'][k])for k,v in heads[seed].state_dict().items())
        values=np.asarray(results[seed]);assert values.shape==(2189,78)
        np.savez_compressed(out/('EVAL_QWEN_STEREO78_seed'+str(seed)+'.npz'),signals=np.asarray([r['signal']for r in records[2:]]),features=values,q39=values[:,:39])
    save('complete.json',dict(rows=2189,seeds=[44,45],training_updates=0,VALID_control=proof,
         public_weights_unchanged=True,seconds=time.monotonic()-start,
         outputs={p.name:env['sha'](p)for p in out.glob('EVAL_*.npz')}))


if __name__=='__main__':main()
