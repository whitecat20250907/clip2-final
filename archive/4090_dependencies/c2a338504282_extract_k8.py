"""Fresh matched pretrained/own-OOF K8 evidence; own outputs, no labels.

Two deterministic worker shards; hash-verified chunk and partial-row resume.
An old frozen cache failed whole-vector replay and is deliberately not used.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import numpy as np
import torch
from public_features import infer_record
from train_adapters import adapters,restore,atomic_json
HERE=Path(__file__).resolve().parent;EXPS=HERE.parent
sys.path.insert(0,str(EXPS/'d6e_split_listener_adapter_probe_v1'))
from audit_targets import read,sha,digest
from lora import attach
STOP=False


def stop(*_):
    global STOP
    STOP=True


def worker_ids(ids,worker):
    assert worker in (0,1)
    return ids[worker::2]


def inputs():
    pf=HERE/'results/oof_plan_v1/plan.json';plan=read(pf)
    assert sha(pf)=='a55be82579339dd6d9cee0925676a230bd67144baf3f4ce7d2573e0109290249'
    rf=EXPS/'d6e_train_validation_reset_v1/results/audit_v1/protocol.json';assert sha(rf)==plan['role_protocol_sha256']
    pubf=EXPS/'d6e_multitrack_word_experts_v1/results/assets_v1/public.json'
    assert sha(pubf)=='7944bca70172a47eb0ace71118fd7d1ed505faf4ebaac92178ad086c1553482d'
    public=read(pubf);training=HERE/'results/adapter_training_v1';done=read(training/'report.json')
    assert done['status']=='three_newFIT_OOF_adapters_training_complete' and len(done['jobs'])==3
    assert sum(j['optimizer_steps'] for j in done['jobs'])==8415
    assert sha(training/'protocol.json')==done['protocol_sha256']
    training_protocol=read(training/'protocol.json')
    for name,h in training_protocol['own_sources'].items():assert sha(HERE/name)==h
    for name,h in training_protocol['own_borrowed_sources'].items():assert sha(EXPS/'d6e_split_listener_adapter_probe_v1'/name)==h
    forbidden=set(plan['validation_ids']+plan['formal_DEV_ids']+plan['original_excluded_train_ids']);seen=set()
    for job in plan['jobs']:
        meta=read(training/(job['name']+'_complete.json'))
        assert meta['base_weights_unchanged'] and meta['protocol_sha256']==done['protocol_sha256']
        assert meta['supervised_ids_sha256']==job['supervised_fit_ids_sha256']
        assert meta['feature_only_ids_sha256']==job['feature_only_ids_sha256']
        assert sha(training/meta['checkpoint'])==meta['checkpoint_sha256']
        assert not set(job['supervised_fit_ids'])&forbidden
        assert not set(job['feature_only_ids'])&set(job['supervised_fit_ids'])
        assert not {plan['group_of'][s] for s in job['feature_only_ids']}&{plan['group_of'][s] for s in job['supervised_fit_ids']}
        assert not seen&set(job['feature_only_ids']);seen.update(job['feature_only_ids'])
    assert seen==set(plan['fit_ids']+plan['validation_ids']) and len(seen)==16204
    modeldir=Path('/home/lisiyi/models/whisper-medium');assert sha(modeldir/'model.safetensors')==plan['model_weight_sha256']
    return plan,public,training,modeldir


def contract(plan,training,modeldir):
    return dict(schema='fresh_public_matched_OOF_K8_v1',plan_sha256=sha(HERE/'results/oof_plan_v1/plan.json'),
        training_report_sha256=sha(training/'report.json'),model_weight_sha256=plan['model_weight_sha256'],
        own_borrowed_source_hashes=read(training/'protocol.json')['own_borrowed_sources'],
        model_behavior_files={p.name:sha(p) for p in modeldir.iterdir() if p.is_file() and p.suffix in ('.json','.txt')},
        public_sha256='7944bca70172a47eb0ace71118fd7d1ed505faf4ebaac92178ad086c1553482d',
        own_sources={n:sha(HERE/n) for n in ['extract_k8.py','public_features.py','train_adapters.py','run_k8_heads.py','tf_readout.py']},
        model_precision='BF16 pretrained base and own FP32 LoRA, SDPA, eval, no autocast',
        generation=dict(K=8,temperature=.7,top_k=0,top_p=1.,num_beams=1,max_new_tokens=128,return_dict_in_generate=False),
        RNG='Fixed SHA256(K8/2042/sid/ear) first4bytes per record/ear, identical both conditions',
        worker_split='Two interleaved public ID shards per OOF job; never label-based',workers=2,chunk_records=32,
        columns=['Better-ear scoring-word TF summed log probability','Fraction of8 generated candidates matching scoring word','Any of8 candidates matches scoring word'],
        ear_selection='Utterance-level ear with greater mean local word match; all scoring words retained',
        earlier_cache_reused=False,base_and_adapted_both_fresh=True,peer_model_or_predictions_used=False,
        response_manifest_opened=False,score_values_read=False,formal4059DEV_disabled=True,official_valid_used=False,
        scorer=dict(seeds=[42,43,44],arms=['context_only','base_listener','adapted_listener','wrong_adapted'],
            epochs=3,batch=32,learning_rate=.0003,weight_decay=.01,width=32,logit_bound=.25,
            fitting_scope='newFIT only',checkpoint='Fixed last epoch, no held-out selection',
            rule='Adapted RMSE beats all4controls in all3seeds, PCC>=parent, mean RMSE gain>=0.0002 vs parent/base/wrong; resource rule, not efficacy proof'),
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        attribution='Borrowed Zhao/Yang listener-answer and simulated-word mechanism, integrated into own word readout; no exclusive novelty claim')


def verify_bank_file(path,entry,selected,ph):
    assert path.is_relative_to(HERE) and sha(path)==entry['sha256']
    assert entry['protocol_sha256']==ph and entry['ids_sha256']==digest(selected)
    with np.load(path,allow_pickle=False) as z:
        assert z['signals'].tolist()==selected
        lengths=z['lengths'];base=z['base_word3'];adapt=z['adapted_word3']
        assert lengths.shape==(len(selected),) and (lengths>0).all()
        assert base.shape==adapt.shape==(int(lengths.sum()),3)
        assert np.isfinite(base).all() and np.isfinite(adapt).all()
        assert (base[:,0]<=1e-6).all() and (adapt[:,0]<=1e-6).all()
        assert ((base[:,1:]>=0)&(base[:,1:]<=1)).all() and ((adapt[:,1:]>=0)&(adapt[:,1:]<=1)).all()
        return lengths.copy(),base.copy(),adapt.copy()


def save_chunk(path,job,worker,ids,lengths,base,adapt,ph):
    temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,signals=np.array(ids),lengths=np.array(lengths,np.int32),
        base_word3=np.concatenate(base),adapted_word3=np.concatenate(adapt))
    os.replace(temp,path)
    entry=dict(job=job,worker=worker,file=str(path),sha256=sha(path),rows=len(ids),words=int(sum(lengths)),ids_sha256=digest(ids),protocol_sha256=ph)
    atomic_json(path.with_suffix('.json'),entry);return entry


def models(modeldir,device):
    from transformers import WhisperForConditionalGeneration,WhisperTokenizerFast,WhisperFeatureExtractor
    base=WhisperForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,use_safetensors=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    adapted=WhisperForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,use_safetensors=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval();assert len(attach(adapted,16))==96
    tokenizer=WhisperTokenizerFast.from_pretrained(modeldir,local_files_only=True,language='english',task='transcribe')
    extractor=WhisperFeatureExtractor.from_pretrained(modeldir,local_files_only=True)
    return base,adapted,tokenizer,extractor


def paired(base,adapted,tokenizer,extractor,public,sid,device):
    import soundfile as sf
    info=public[sid]['files']['audio1'];path=Path(info['path']);assert sha(path)==info['sha256']
    audio,rate=sf.read(path,dtype='float32',always_2d=True)
    b=infer_record(base,tokenizer,extractor,audio,rate,sid,public[sid]['prompt'],device)
    a=infer_record(adapted,tokenizer,extractor,audio,rate,sid,public[sid]['prompt'],device)
    assert b['word3'].shape==a['word3'].shape==(public[sid]['n_words'],3)
    return b['word3'],a['word3']


def probe(plan,public,training,modeldir,protocol):
    out=HERE/'results/fresh_K8_dual_model_probe_v1';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'report.json').exists(),'Preserve completed probe'
    atomic_json(out/'protocol.json',protocol);torch.set_num_threads(4);device=torch.device('cuda:0')
    base,adapted,tokenizer,extractor=models(modeldir,device);initial=adapters(adapted)
    job=plan['jobs'][0];selected=sorted(job['feature_only_ids'],key=lambda s:digest(['fresh-K8-runtime-public-v1',s]))[:16]
    state=torch.load(training/(job['name']+'_final.pt'),map_location='cpu',weights_only=True)['adapters']
    seconds=[];zero_gap=0.;torch.cuda.reset_peak_memory_stats()
    for i,sid in enumerate(selected):
        if i<2:
            restore(adapted,initial);b,z=paired(base,adapted,tokenizer,extractor,public,sid,device)
            zero_gap=max(zero_gap,float(np.abs(b-z).max()));assert zero_gap<1e-5,'Zero-adapter versus plain base changed public inference'
        restore(adapted,state);torch.cuda.synchronize();tick=time.perf_counter()
        b,a=paired(base,adapted,tokenizer,extractor,public,sid,device);torch.cuda.synchronize();seconds.append(time.perf_counter()-tick)
        print('FRESH_K8_PROBE_ROW '+json.dumps(dict(done=i+1,seconds=seconds[-1])),flush=True)
    free,total=torch.cuda.mem_get_info();peak=torch.cuda.max_memory_allocated()/2**30
    report=dict(status='fresh_dual_model_K8_FIT_runtime_probe_complete',public_FIT_ids=selected,zero_adapter_max_gap=zero_gap,
        protocol_sha256=digest(protocol),post_warmup_seconds_per_record=float(np.mean(seconds[2:])),peak_GPU_allocated_gib=peak,free_GPU_gib=free/2**30,
        estimated_two_worker_minutes=float(np.mean(seconds[2:]))*16204/120,resource_gate_passed=bool(peak<=8 and free/2**30>=2),
        answer_or_score_values_read=False,optimizer_steps=0,features_saved=False,CLIP2_improvement_measured=False,
        limitation='Sixteen FIT public inputs; parallel shared-GPU/storage variability and setup costs remain')
    atomic_json(out/'report.json',report);print('FRESH_K8_RUNTIME_COMPLETE '+json.dumps(report),flush=True)


def export_worker(root,worker,plan,public,training,modeldir,protocol):
    pf=root/'protocol.json';assert read(pf)==protocol;ph=sha(pf)
    probe_report=read(HERE/'results/fresh_K8_dual_model_probe_v1/report.json')
    assert probe_report['resource_gate_passed'] and probe_report['zero_adapter_max_gap']<1e-5
    assert probe_report['protocol_sha256']==digest(protocol)
    lock=root/f'worker{worker}.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),host=socket.gethostname()),f)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    receipts=[];tick=time.perf_counter()
    try:
        if (root/f'worker{worker}_complete.json').exists():print('K8_WORKER_ALREADY_COMPLETE',worker,flush=True);return
        torch.set_num_threads(4);device=torch.device('cuda:0');base,adapted,tokenizer,extractor=models(modeldir,device)
        for job in plan['jobs']:
            name=job['name'];folder=root/name;folder.mkdir(exist_ok=True);ids=worker_ids(job['feature_only_ids'],worker)
            restore(adapted,torch.load(training/(name+'_final.pt'),map_location='cpu',weights_only=True)['adapters'])
            for start in range(0,len(ids),32):
                selected=ids[start:start+32];path=folder/f'worker{worker}_chunk_{start:05d}.npz';mp=path.with_suffix('.json')
                if mp.exists():
                    old=read(mp);verify_bank_file(path,old,selected,ph);receipts.append(old);continue
                partial=folder/f'worker{worker}_chunk_{start:05d}.partial.npz';partialmeta=partial.with_suffix('.json')
                doneids=[];lengths=[];bs=[];aas=[]
                if partialmeta.exists():
                    old=read(partialmeta);assert 0<old['rows']<len(selected)
                    doneids=selected[:old['rows']];ls,b,a=verify_bank_file(partial,old,doneids,ph);lengths=ls.tolist()
                    offsets=np.concatenate(([0],np.cumsum(ls)));bs=[b[offsets[i]:offsets[i+1]] for i in range(len(ls))];aas=[a[offsets[i]:offsets[i+1]] for i in range(len(ls))]
                try:
                    for sid in selected[len(doneids):]:
                        if STOP:raise InterruptedError('Stop requested; completed public rows retained')
                        b,a=paired(base,adapted,tokenizer,extractor,public,sid,device)
                        doneids.append(sid);lengths.append(public[sid]['n_words']);bs.append(b);aas.append(a)
                except BaseException:
                    if doneids:save_chunk(partial,name,worker,doneids,lengths,bs,aas,ph)
                    raise
                entry=save_chunk(path,name,worker,doneids,lengths,bs,aas,ph);receipts.append(entry)
                if partialmeta.exists():
                    previous=read(partialmeta);assert sha(partial)==previous['sha256'];partial.unlink();partialmeta.unlink()
                atomic_json(root/f'worker{worker}_progress.json',dict(pid=os.getpid(),job=name,worker=worker,done=start+len(selected),total=len(ids),seconds=time.perf_counter()-tick,completed_chunks=len(receipts)))
                print('FRESH_K8_CHUNK_COMPLETE '+json.dumps(dict(job=name,worker=worker,done=start+len(selected),total=len(ids),seconds=round(time.perf_counter()-tick,2))),flush=True)
        atomic_json(root/f'worker{worker}_complete.json',dict(status='fresh_K8_worker_complete',worker=worker,protocol_sha256=ph,receipts=receipts,
            rows=sum(e['rows'] for e in receipts),seconds=time.perf_counter()-tick,response_values_read=False,score_values_read=False,formal_DEV_evaluated=False))
    except BaseException as e:
        atomic_json(root/f'worker{worker}_last_failure.json',dict(type=type(e).__name__,message=str(e),completed_chunks=len(receipts),stop_requested=STOP))
        raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()


def assemble(root,plan):
    ph=sha(root/'protocol.json');receipts=[];seen=set()
    for worker in (0,1):
        r=read(root/f'worker{worker}_complete.json');assert r['status']=='fresh_K8_worker_complete' and r['protocol_sha256']==ph
        for entry in r['receipts']:
            job=next(j for j in plan['jobs'] if j['name']==entry['job']);ids=worker_ids(job['feature_only_ids'],worker)
            path=Path(entry['file']);assert path.parent==root/job['name'] and entry['worker']==worker
            with np.load(path,allow_pickle=False) as z:selected=z['signals'].tolist()
            assert set(selected)<=set(ids) and not seen&set(selected);verify_bank_file(path,entry,selected,ph)
            seen.update(selected);receipts.append(entry)
    assert seen==set(plan['fit_ids']+plan['validation_ids']) and len(seen)==16204
    report=dict(status='public_OOF_K8_features_complete',protocol_sha256=ph,receipts=receipts,rows=16204,
        base_and_adapted_fresh=True,old_frozen_cache_used=False,response_values_read=False,score_values_read=False,formal_DEV_evaluated=False,
        limitation='Two-worker inference only; no intelligibility improvement measured')
    atomic_json(root/'report.json',report);print('FRESH_K8_FEATURES_COMPLETE '+json.dumps({k:v for k,v in report.items() if k!='receipts'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','probe','export','assemble'],required=True);p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    assert str(HERE).startswith('/home/lisiyi/clip2_lmy/CLIP2_lmy/my_experiments/')
    plan,public,training,modeldir=inputs();protocol=contract(plan,training,modeldir);root=HERE/'results/k8_features_v1';root.mkdir(parents=True,exist_ok=True)
    if a.mode=='freeze':
        pf=root/'protocol.json'
        if pf.exists():assert read(pf)==protocol
        else:atomic_json(pf,protocol)
        print('FRESH_K8_PROTOCOL_FROZEN '+sha(pf),flush=True)
    elif a.mode=='probe':probe(plan,public,training,modeldir,protocol)
    elif a.mode=='export':assert a.worker is not None;export_worker(root,a.worker,plan,public,training,modeldir,protocol)
    else:assert read(root/'protocol.json')==protocol;assemble(root,plan)


if __name__=='__main__':main()
