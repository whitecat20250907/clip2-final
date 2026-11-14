"""Original full model inference per condition; identical contraction proxy.

Uses only public audio/reference text. No listener answers at inference. The
proxy is not the full official spelling/homophone/BEEP alternatives pipeline.
Own frozen upstream functions/modules are supplied by the hash-checking runner.
"""
import hashlib
import numpy as np
import torch


def matched_candidates(public_fn,label_fn,contractions,prompt,texts,expanded):
    normalized=[public_fn.normalize(t) for t in texts]
    if expanded:normalized=[' '.join(label_fn.expanded_words(t,contractions)) for t in normalized]
    assert len(normalized)==8 and all(len(t.split())<=300 for t in normalized)
    return np.stack([public_fn.aligned_hits(prompt,t) for t in normalized])


def aggregate(teacher,hit_vectors):
    # Preserve the original two-stage float32 reduction, including ties.
    selected=int(np.argmax([h.mean(1).mean() for h in hit_vectors]))
    word=np.stack([teacher[selected],hit_vectors[selected].mean(0),hit_vectors[selected].max(0)],axis=-1)
    both=np.stack([h.mean(1) for h in hit_vectors]);best=both[selected];paired=both.mean(0)
    summary=np.array([best.mean(),best.std(),(best==0).mean(),paired.mean(),paired.std(),(paired==0).mean()],np.float32)
    return dict(word3=word,sentence6=summary,selected_ear=selected)


@torch.no_grad()
def paired_full_inference(base,adapted,tokenizer,extractor,audio,rate,sid,prompt,device,
                          public_fn,label_fn,contractions):
    import soxr
    ids,owners,n=public_fn.mapping(tokenizer,prompt)
    assert audio.ndim==2 and audio.shape[1]==2 and 0<len(audio)/rate<30 and np.isfinite(audio).all()
    teachers={k:[] for k in ['base','adapted']};texts={k:[] for k in teachers}
    for ear in (0,1):
        wave=soxr.resample(audio[:,ear],rate,16000,quality='HQ')
        inputs=extractor(wave,sampling_rate=16000,return_attention_mask=True,return_tensors='pt')
        f=inputs.input_features.to(device=device,dtype=torch.bfloat16);mask=inputs.attention_mask.to(device)
        decoder=torch.tensor([ids[:-1]],device=device);expected=torch.tensor(ids[1:],device=device)[:,None]
        for name,model in [('base',base),('adapted',adapted)]:
            result=model(input_features=f,attention_mask=mask,decoder_input_ids=decoder,use_cache=False)
            lp=result.logits.float().log_softmax(-1)[0].gather(1,expected)[:,0].cpu().numpy()
            teachers[name].append(np.array([lp[owners==i].sum() for i in range(n)],np.float32))
            seed=int.from_bytes(hashlib.sha256(f'K8/2042/{sid}/{ear}'.encode()).digest()[:4],'little')
            # Keep the original complete generation path, including Whisper's
            # internal segment/retry behavior. Encoder sharing was rejected after
            # a public record took a different path than the bounded probe.
            with torch.random.fork_rng(devices=[device.index or 0]):
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                generated=model.generate(input_features=f,attention_mask=mask,
                    language='english',task='transcribe',return_timestamps=False,
                    do_sample=True,temperature=.7,top_k=0,top_p=1.,num_beams=1,
                    num_return_sequences=8,max_new_tokens=128,begin_suppress_tokens=[],
                    return_dict_in_generate=False)
            texts[name].append(tokenizer.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False))
    output={}
    for name in teachers:
        original=[matched_candidates(public_fn,label_fn,contractions,prompt,t,False) for t in texts[name]]
        expanded=[matched_candidates(public_fn,label_fn,contractions,prompt,t,True) for t in texts[name]]
        output[name]=dict(original=aggregate(teachers[name],original),normalized=aggregate(teachers[name],expanded),
            teacher_by_ear=np.stack(teachers[name]),texts_by_ear=texts[name])
    return output
