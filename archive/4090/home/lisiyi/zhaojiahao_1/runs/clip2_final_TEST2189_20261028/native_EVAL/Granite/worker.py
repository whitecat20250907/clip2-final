"""Native vocabulary-space observations and full time-support qualification."""
from pathlib import Path
import argparse,os,sys,json,time,re,hashlib,traceback
ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--role',required=True);a=ap.parse_args();b=Path(a.base);p=json.loads((b/'protocol.json').read_bytes());assets=Path(p['assets']);assert b.stat().st_uid==os.getuid()==p['owner_uid'];sys.path.insert(0,str(assets/'site'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(b/'private_cache/hf'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
import numpy as np,torch,soundfile as sf,soxr
from transformers import AutoProcessor,AutoModelForCTC,ParakeetForTDT
from v10_ER_temporal_math_20261022 import ctc_profile,tdt_profile,math_qualification
torch.set_num_threads(2);out=b/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic();model_name=a.role.split('_')[0];part=int(a.role.rsplit('_',1)[1]);rows=json.loads((b/'selection.json').read_bytes())[part::p['parts']];modelpath=Path(p.get('model_paths',{}).get(model_name,str(assets/'models'/model_name)))

def save(name,obj):
    q=out/(name+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2));q.replace(out/name)

def word_map(tokenizer,text):
    enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];spans=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)];owners=[]
    for s,e in enc['offset_mapping']:
        matches=[i for i,(lo,hi) in enumerate(spans) if any(not text[j].isspace() for j in range(max(s,lo),min(e,hi)))]
        owners.append(matches[0] if len(matches)==1 else -1)
    assert len(ids)==len(owners) and len(ids)>0
    return ids,owners,spans

def profiles(tokenprof,owners,n):
    result=np.zeros((n,tokenprof.shape[1]),dtype=np.float64);valid=np.zeros(n,dtype=bool)
    for i in range(n):
        ix=[k for k,v in enumerate(owners) if v==i]
        if ix:result[i]=tokenprof[ix].sum(0);result[i]/=result[i].sum();valid[i]=True
    return result.astype('float32'),valid

def audio_file(sid,source,split='train'):
    root=Path(p['data_root']);tail=('signals',sid+'.flac') if source=='A1' else ('unprocessed',sid+'_unproc.flac')
    for candidate in [root/split/tail[0]/tail[1],root/'audio'/split/tail[0]/tail[1]]:
        if candidate.is_file():return candidate
    raise FileNotFoundError(str(root/split/tail[0]/tail[1]))

def infer(wave,ids,owners,nwords,do_replay=False):
    inp=processor(wave,sampling_rate=16000,return_tensors='pt');inp={k:v.to('cuda') if hasattr(v,'to') else v for k,v in inp.items()}
    if model_name=='Granite':
        if 'input_features' in inp:inp['input_features']=inp['input_features'].to(torch.bfloat16)
        captured=[];hook=model.encoder.out.register_forward_hook(lambda module,args,output:captured.append(output.detach().float().cpu()))
        try:
            with torch.inference_mode():final=model(**inp).logits.detach().float().cpu()
        finally:hook.remove()
        assert len(captured)==1,(len(captured),'native midpoint hook count')
        if do_replay:
            with torch.inference_mode():again=model(**inp).logits.detach().float().cpu()
            err=float((again-final).abs().max());assert err==0,err
            save('native_hook_parity.json',dict(max_logit_difference=err,native_mid_calls=1,forward_modified=False))
        outputs={};meta={}
        for key,logits in [('final',final),('mid',captured[0])]:
            lp=logits[0].log_softmax(-1).numpy().astype('float64');tp,inf=ctc_profile(lp,ids,int(model.config.pad_token_id));wp,valid=profiles(tp,owners,nwords);outputs[key]=wp
            support=(tp*lp[:,ids].T).sum(1);inf['token_native_logp_weighted_mean']=support.tolist();inf['word_native_logp_mean']=[float(support[np.asarray(owners)==i].mean()) if i in owners else None for i in range(nwords)]
            greedy=[];prev=None
            for token in np.argmax(lp,axis=-1):
                if token!=prev and token!=int(model.config.pad_token_id):greedy.append(int(token))
                prev=token
            inf['greedy_text']=processor.tokenizer.decode(greedy,skip_special_tokens=True);meta[key]=inf
        frame_hop=float(getattr(processor.feature_extractor,'hop_length',160))/16000*int(getattr(processor.feature_extractor,'frame_stacking',1))*2**len(model.config.encoder_config.subsample_layers)
    else:
        with torch.inference_mode():
            encout=model.get_audio_features(**inp,output_attention_mask=True);enc=encout.pooler_output
            if encout.attention_mask is not None:
                length=int(encout.attention_mask[0].sum());assert bool(encout.attention_mask[0,:length].all()) and not bool(encout.attention_mask[0,length:].any());enc=enc[:,:length]
            assert bool(torch.isfinite(enc).all()),'Nonfinite valid Parakeet encoder frame'
            dec=model.decoder(torch.tensor([[int(model.config.blank_token_id)]+ids],device='cuda'));assert bool(torch.isfinite(dec).all())
            U=len(ids);bl=[];nx=[];du=[];v=int(model.config.vocab_size)
            for t in range(0,enc.shape[1],8):
                logits=model.joint(encoder_hidden_states=enc[:,t:t+8,None,:],decoder_hidden_states=dec[:,None,:,:])[0].float();tl=logits[:,:,:v].log_softmax(-1);dl=logits[:,:,v:].log_softmax(-1)
                bl.append(tl[:,:,int(model.config.blank_token_id)].cpu().numpy());nx.append(tl[:,:U].gather(-1,torch.tensor(ids,device='cuda')[None,:,None].expand(tl.shape[0],-1,1)).squeeze(-1).cpu().numpy());du.append(dl.cpu().numpy())
        next_lp=np.concatenate(nx).astype('float64');tp,inf=tdt_profile(np.concatenate(bl).astype('float64'),next_lp,np.concatenate(du).astype('float64'),list(model.config.durations));wp,valid=profiles(tp,owners,nwords);outputs={'final':wp}
        support=(tp*next_lp.T).sum(1);inf['token_native_logp_weighted_mean']=support.tolist();inf['word_native_logp_mean']=[float(support[np.asarray(owners)==i].mean()) if i in owners else None for i in range(nwords)];meta={'final':inf}
        frame_hop=float(getattr(processor.feature_extractor,'hop_length',160))/16000*int(model.config.encoder_config.subsampling_factor);meta['final']['valid_encoder_frames']=int(enc.shape[1]);meta['final']['padded_encoder_frames_discarded']=int(encout.pooler_output.shape[1]-enc.shape[1])
    meta.update(frame_hop_seconds=frame_hop,time_origin='frame index*hop +hop/2 approximate center; frontend receptive-field offset not yet calibrated',word_coverage=int(valid.sum()),words=nwords,unmapped_tokens=sum(v<0 for v in owners),reference_conditional=True,word_existence_claim=False)
    return outputs,valid,meta

try:
    save('math_qualification.json',math_qualification());processor=AutoProcessor.from_pretrained(str(modelpath),local_files_only=True)
    factory=AutoModelForCTC if model_name=='Granite' else ParakeetForTDT;dtype=torch.bfloat16 if model_name=='Granite' else torch.float32
    backend='eager' if model_name=='Granite' else 'sdpa'
    model=factory.from_pretrained(str(modelpath),local_files_only=True,torch_dtype=dtype,attn_implementation=backend).cuda().eval().requires_grad_(False);save('model_loaded.json',dict(model=model_name,seconds=time.monotonic()-start,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,dtype=str(dtype),attention_backend=backend,public_snapshot=str(modelpath)))
    finished=[]
    for ri,row in enumerate(rows):
        sid=row['signal_id'];ids,owners,spans=word_map(processor.tokenizer,row['prompt']);record=dict(signal_id=sid,prompt=row['prompt'],severity=row['severity'],token_ids=ids,token_word_owners=owners,word_spans=spans,streams={});arrays={};firstwave=None
        for source in ['A1','A2']:
            path=audio_file(sid,source,row.get('split','train'));audio,sr=sf.read(path,dtype='float32',always_2d=True)
            if sr!=16000:audio=soxr.resample(audio,sr,16000)
            for ear,ch in [('L',0),('R',min(1,audio.shape[1]-1))]:
                key=source+ear;wave=np.ascontiguousarray(audio[:,ch]);sha=hashlib.sha256(wave.tobytes()).hexdigest();outputs,valid,meta=infer(wave,ids,owners,len(spans),do_replay=ri==0 and key=='A1L');meta.update(PCM_SHA=sha,duration_seconds=len(wave)/16000,source=str(path));record['streams'][key]=meta;arrays[key+'_valid']=valid
                for stage,wp in outputs.items():arrays[key+'_'+stage]=wp
                if firstwave is None:firstwave=wave
                save('progress.json',dict(model=model_name,part=part,rows_done=len(finished),rows_total=len(rows),current=sid,stream=key,seconds=time.monotonic()-start))
        if ri==0:
            same,_,_=infer(firstwave,ids,owners,len(spans));zero,_,zm=infer(np.zeros_like(firstwave),ids,owners,len(spans));reference=arrays['A1L_final'];sameerr=float(np.max(np.abs(same['final']-reference)));assert sameerr<1e-6,sameerr
            record['controls']=dict(same_audio_profile_max_error=sameerr,silence=zm,warning='Reference conditioning can force a localized path even on silence; logZ and missing flags must accompany shape features')
            arrays['silence_final']=zero['final']
            if p.get('time_shift_control'):
                shifted,_,sm=infer(np.pad(firstwave,(5120,0)),ids,owners,len(spans));hop=record['streams']['A1L']['frame_hop_seconds'];centers=(np.arange(reference.shape[1])+.5)*hop;newcenters=(np.arange(shifted['final'].shape[1])+.5)*sm['frame_hop_seconds'];delta=(shifted['final']*newcenters).sum(1)-(reference*centers).sum(1);record['controls']['time_shift']=dict(added_front_silence_seconds=.32,per_word_centroid_shift=delta.tolist(),median_centroid_shift=float(np.median(delta)),caveat='Native-context response diagnostic, not human correctness; time-origin convention remains explicitly recorded');arrays['padded_front_final']=shifted['final']
        np.savez_compressed(out/(sid+'.npz'),**arrays);save(sid+'.json',record);finished.append(sid);print(json.dumps(dict(done=sid,model=model_name,rows=len(finished),seconds=time.monotonic()-start)),flush=True)
    save('complete.json',dict(status='native_time_profiles_extracted',model=model_name,part=part,rows=finished,streams_per_row=4,midpoint_native=model_name=='Granite',seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,TRAIN_score_gate=False,not_official_candidate_yet=True,qualification_only=p.get('qualification_only',True)))
except Exception as exc:
    save('failure.json',dict(error=repr(exc),signal_id=locals().get('sid'),stream=locals().get('key'),trace=traceback.format_exc()[-7000:],seconds=time.monotonic()-start));raise
