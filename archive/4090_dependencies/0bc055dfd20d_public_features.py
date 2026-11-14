"""Public-only listener inference, attributed to audited Zhao/Yang K8 recipe.

Own implementation: never imports peer code or reads human responses.
Levenshtein matches are an ASR surrogate, not the official scoring pipeline.
"""
import hashlib
import re
import unicodedata
import numpy as np
import torch


def normalize(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=text.translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"'}))
    text=re.sub(r'[-–—/]',' ',text)
    return ' '.join(re.sub(r"[^\w\s']",'',text).split())


def aligned_hits(reference,hypothesis):
    ref=reference.casefold().split(); hyp=hypothesis.casefold().split()
    n,m=len(ref),len(hyp); dp=np.zeros((n+1,m+1),dtype=np.int32)
    dp[:,0]=np.arange(n+1); dp[0]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i,j]=min(dp[i-1,j-1]+(ref[i-1]!=hyp[j-1]),dp[i-1,j]+1,dp[i,j-1]+1)
    result=np.zeros(n,np.float32); i,j=n,m
    while i or j:
        if i and j and dp[i,j]==dp[i-1,j-1]+(ref[i-1]!=hyp[j-1]):
            result[i-1]=ref[i-1]==hyp[j-1]; i-=1; j-=1
        elif i and dp[i,j]==dp[i-1,j]+1: i-=1
        else: j-=1
    return result


def mapping(tokenizer,text):
    assert normalize(text)==' '.join(text.casefold().split()),'Scoring text not normalized'
    encoded=tokenizer(text,add_special_tokens=True,return_offsets_mapping=True)
    spans=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
    owners=[]
    for left,right in encoded['offset_mapping'][1:]:
        touched=[i for i,(a,b) in enumerate(spans) if max(a,left)<min(b,right)]
        assert len(touched)<=1,'Tokenizer token crosses scoring words'
        owners.append(touched[0] if touched else -1)
    assert set(owners)-{-1}==set(range(len(spans)))
    assert len(encoded['input_ids'])<=448
    return encoded['input_ids'],np.array(owners),len(spans)


@torch.no_grad()
def infer_record(model,tokenizer,extractor,audio,rate,sid,prompt,device):
    import soxr
    ids,owners,n=mapping(tokenizer,prompt)
    assert audio.ndim==2 and audio.shape[1]==2 and 0<len(audio)/rate<30
    assert np.isfinite(audio).all()
    word=[]; candidate_scores=[]; missing_eos=0
    for ear in (0,1):
        wave=soxr.resample(audio[:,ear],rate,16000,quality='HQ')
        inputs=extractor(wave,sampling_rate=16000,return_attention_mask=True,return_tensors='pt')
        features=inputs.input_features.to(device=device,dtype=torch.bfloat16)
        attention=inputs.attention_mask.to(device)
        decoder=torch.tensor([ids[:-1]],device=device)
        result=model(input_features=features,attention_mask=attention,decoder_input_ids=decoder,use_cache=False)
        lp=result.logits.float().log_softmax(-1)[0].gather(1,torch.tensor(ids[1:],device=device)[:,None])[:,0].cpu().numpy()
        summed=np.array([lp[owners==i].sum() for i in range(n)],np.float32)
        seed=int.from_bytes(hashlib.sha256(f'K8/2042/{sid}/{ear}'.encode()).digest()[:4],'little')
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            generated=model.generate(input_features=features,attention_mask=attention,
                language='english',task='transcribe',return_timestamps=False,
                do_sample=True,temperature=.7,top_k=0,top_p=1.,num_beams=1,
                num_return_sequences=8,max_new_tokens=128,begin_suppress_tokens=[],
                return_dict_in_generate=False)
        missing_eos+=sum(50257 not in x.tolist() for x in generated)
        texts=tokenizer.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        hits=np.stack([aligned_hits(prompt,normalize(t)) for t in texts])
        assert hits.shape==(8,n)
        word.append(np.stack([summed,hits.mean(0),hits.max(0)],axis=-1))
        candidate_scores.append(hits.mean(1))
    selected=int(np.argmax([x.mean() for x in candidate_scores]))
    both=np.stack(candidate_scores); best=both[selected]; paired=both.mean(0)
    summary=np.array([best.mean(),best.std(),(best==0).mean(),paired.mean(),paired.std(),(paired==0).mean()],np.float32)
    return dict(word3=word[selected],sentence6=summary,selected_ear=selected,sequences_without_eos=missing_eos)
