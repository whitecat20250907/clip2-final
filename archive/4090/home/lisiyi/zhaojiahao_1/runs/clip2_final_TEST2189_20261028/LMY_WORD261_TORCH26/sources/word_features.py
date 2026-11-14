"""Exact frozen matching recipe; reuse frames, write compact valid words only."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from contracts import read, sha, digest, require, freeze, save_arrays, emit
import dependencies as dep


def pronunciation(row, public, m, transfer, dictionary, contractions):
    try:
        words = transfer.normalized_words(row['original_prompt'])
        parts = [m.word_phones(w,dictionary) for w in words]
        tokens = [p for block in parts for p in block]
        require(1 <= len(tokens) <= 200, 'Original phone_count_outside_1_to_200')
        return transfer.query(dict(row,query_id='valid'), public,
            {'valid':dict(tokens=tokens)},dictionary,contractions,m)
    except ValueError as exc:
        return dict(phones=[],boundaries=[],phone_to_word=[],reason=str(exc),mapped_words=0,
                    sung_words=None,scoring_words=public['n_words'])


@torch.no_grad()
def one(source, sid, fast, core):
    q=source.queries[sid]; w=source.public[sid]['n_words']
    if not q['phones'] or not q['mapped_words']:
        return np.zeros((w,261),np.float16),np.zeros(w,bool),dict(signal=sid,masked=True,reason=q['reason'])
    heard,a1=source.audio1(sid); reference,a2=source.audio2(sid)
    h=torch.tensor(heard,device=source.device)[None]; r=torch.tensor(reference,device=source.device)[None]
    hl=torch.tensor([h.shape[1]],device=source.device); rl=torch.tensor([r.shape[1]],device=source.device)
    ids=torch.tensor(q['phones'],device=source.device)[None]
    bl=torch.tensor(q['boundaries'],device=source.device)[None]; pl=torch.tensor([ids.shape[1]],device=source.device)
    model=source.matchers['true_pair']
    ar=model.encode_audio(r,rl)[0]; ah=model.encode_audio(h,hl)[0]; text=model.encode_phones(ids,bl,pl)[0]
    post,_,diag=fast.fast_posterior(ar@text.T)
    rs,hs=(source.public[sid]['files'][k]['seconds'] for k in ('audio2','audio1'))
    v,k=core.phone_evidence(text,ar,ah,post,rs,hs)
    v,k=core.word_evidence(v,q['phone_to_word'],k,w)
    # Match the training bank's fp16 storage, followed by fp32 head inputs.
    value=v.cpu().numpy().astype(np.float16); mask=k.cpu().numpy()
    require(value.shape==(w,261) and mask.shape==(w,) and np.isfinite(value).all(), 'Invalid compact evidence')
    return value,mask,dict(signal=sid,Audio1=a1,Audio2=a2,posterior=diag)


def prepare(args, frozen, rows, pooled, device, out, helpers):
    u,core=helpers
    full=dep.module(dep.EXPS/'d6e_grounded_word_score_v1/full_inputs.py','_valid_original_full_inputs')
    transfer=full.api()
    m,p,oldpublic,oldrows,oldqueries,probe_ids,_=transfer.prepare(args)
    require(p==frozen.p, 'Frame/mapping source differs from frozen bundle')
    dictionary=args.dictionary
    if not dictionary.is_file(): dictionary=dictionary.parent/'transfer_lexicon/cmudict.dict'
    lex=m.lexicon(dictionary)
    # New queries must reproduce the recorded TRAIN/DEV pronunciation policy,
    # including whole-query masking, <=200 phones and vocabulary rejection.
    for sid,q in oldqueries.items():
        actual=pronunciation(oldrows[sid],oldpublic[sid],m,transfer,lex,p['contractions'])
        for key in ('phones','boundaries','phone_to_word','mapped_words'):
            require(actual[key]==q[key], 'Pronunciation recipe replay differs: '+sid+' '+key)
    final=read(args.cache/'protocol.json'); prefix=read(args.cache.parent/'phone_memory_prefix_cache_v1/protocol.json')
    require(final['schema']=='phone_memory_valid_frozen_l32_v1' and final['encoder_frozen']
            and final['trim_after_full_context'] and final['labels_used'] is False
            and digest(prefix)==final['prefix_protocol_sha256'] and prefix['plan_sha256']==p['plan_sha256'],
            'Existing Audio2 frame lineage changed')
    owner=args.cache.parent.parent
    require(sha(owner/'frozen_runtime.py')==final['implementation_sha256'], 'Audio2 runtime changed')
    for n,h in prefix['code_sha256'].items():
        require(Path(n).name==n and sha(owner/n)==h, 'Prefix dependency changed')
    require(sha(Path(p['source'])/'results/clip2_full_v1/feature_protocol.json')==prefix['feature_protocol_sha256'],
            'Original Whisper coordinate system changed')
    lib=Path(p['cache_loader']); require(sha(lib)==p['cache_loader_sha256'], 'Original frame loader changed')
    loader=dep.original_loader(lib)
    public={}; queries={}; reused=0
    for i,row in enumerate(rows):
        sid=row['signal']; words=dep.expanded_words(row['prompt'],p['contractions'])
        require(len(words)==row['n_words'], 'Scoring word count differs: '+sid)
        ap=out/'audio_audit'/(sid+'.json')
        files={}
        for role,relative in [('audio1',Path('valid/signals')/(sid+'.flac')),
                              ('audio2',Path('valid/unprocessed')/(sid+'_unproc.flac'))]:
            path=(args.data/relative).resolve()
            require(path.is_file(), 'Missing existing '+role+' '+str(path))
            # Rehash on every resumed run: stat alone is not content verification.
            info=dep.audio_identity(path)
            require(info['channels']==2 and 0<info['seconds']<=30, 'Expected original stereo <=30s audio')
            files[role]=info
        freeze(ap,dict(signal=sid,files=files))
        public[sid]=dict(signal=sid,prompt=row['prompt'],original_prompt=row['original_prompt'],
                        words=words,n_words=len(words),files=files)
        queries[sid]=pronunciation(dict(row,files=files),public[sid],m,transfer,lex,p['contractions'])
        key=digest(['full-context-prefix30-v1',files['audio2']['pcm_sha256'],0])
        reused+=int((args.cache/key[:2]/(key+'.json')).is_file())
        if (i+1)%100==0 or i+1==len(rows): emit('POOLED_VALID_AUDIO_AUDIT',dict(done=i+1,total=len(rows)))
    identity=dict(public_sha256=digest(public),mapping_sha256=digest(queries),dictionary_sha256=sha(dictionary),
        audio2_cache_protocol_sha256=digest(final),prefix_protocol_sha256=digest(prefix),
        original_frame_loader_sha256=sha(lib),valid_frames_root=str(args.valid_frames.resolve()),
        recipe='true_pair/ordered Audio2 alignment/50Hz physical transfer/fp16 word bank',
        valid_id_sha256=digest([r['signal'] for r in rows]))
    freeze(out/'word_extraction_identity.json',identity)
    # Bind exact previously delivered encoder/cache methods. They verify all
    # frozen weights, versions and an old FIT Audio2 before any missing encoding.
    class ValidSource(full.Source):
        def audio1(self,sid):
            if sid not in self.valid_index: return super().audio1(sid)
            x=self.loader(str(args.valid_frames),sid,cache_format='npy_single',feature_dim=1280,requested_layers=[-1])[-1]
            if hasattr(x,'detach'): x=x.detach().cpu().float().numpy()
            x=np.asarray(x,np.float32); require(x.shape==(1500,1280) and np.isfinite(x).all(), 'Invalid valid Audio1 frames')
            gap=float(abs(x.mean(0)-pooled[self.valid_index[sid]]).max())
            require(gap<=1e-4, 'Valid Audio1 cannot replay pooled feature: '+sid)
            n=math.ceil(self.public[sid]['files']['audio1']['seconds']/.02)
            import hashlib
            return x[:n].copy(),dict(signal=sid,full_frame_sha256=hashlib.sha256(x.tobytes()).hexdigest(),pooled_gap=gap)
    s=ValidSource.__new__(ValidSource)
    s.args=args; s.device=device; s.transfer=transfer; s.m=m; s.p=p
    s.public={**oldpublic,**public}; s.queries={**oldqueries,**queries}; s.probe_ids=probe_ids
    s.final=final; s.prefix=prefix; s.loader=loader; s.encoder=None; s.extractor=None
    s.encoded_missing=0; s.encoder_replay=None
    s.valid_index={r['signal']:i for i,r in enumerate(rows)}
    # Needed by inherited Audio1 for the bounded FIT word-feature replay.
    s.index=frozen.index
    with np.load(args.assets/'raw.npz',allow_pickle=False) as z: s.raw={'pooled':z['pooled'].copy()}
    s.matchers=transfer.load_matchers(args,m,device)
    s.matching_state_hashes={a:transfer.state_hash(v.state_dict()) for a,v in s.matchers.items()}
    emit('POOLED_VALID_WORD_READY',dict(records=len(rows),audio2_cache_candidates=reused,
        missing_cache_candidates=len(rows)-reused,encode_missing_audio2=args.encode_missing_audio2,
        masked_queries=sum(not q['phones'] or not q['mapped_words'] for q in queries.values()),
        new_full_frame_cache_written=False,optimizer_steps=0))
    fast=dep.module(dep.EXPS/'d6e_grounded_word_score_v1/fast_path.py','_valid_pooled_fast')
    return s,fast,identity


def replay_words(source, fast, core, bank, ids, out):
    maxgap=0.; quantization_bound=0.
    for sid in ids:
        value,mask,_=one(source,sid,fast,core)
        old,keep,words=bank.batch([sid],0)
        old=old[0,:len(value)].cpu().numpy(); keep=keep[0,:len(value)].cpu().numpy()
        require(np.array_equal(mask,keep), 'Fresh extraction support differs from original word cache')
        # Original fast replay accepts <=2e-5 before fp16 storage. At fp16
        # rounding boundaries that can change a value by one representable step.
        tolerance=np.maximum(2e-5,np.spacing(np.abs(old).astype(np.float16)).astype(np.float32))
        gap=abs(value.astype(np.float32)-old)
        require((gap<=tolerance+2e-5).all(), 'Fresh word recipe differs from original compact cache: '+sid)
        maxgap=max(maxgap,float(gap.max())); quantization_bound=max(quantization_bound,float(tolerance.max()))
    report=dict(records=len(ids),max_abs_gap=maxgap,fp16_spacing_bound=quantization_bound,passed=True,
                score_targets_used=False,selected_from_fixed_FIT_probe=True)
    freeze(out/'word_recipe_replay.json',report); emit('POOLED_VALID_WORD_REPLAY',report)


def build(source, fast, core, rows, out, protocol):
    tick=time.perf_counter(); parts=[]; root=out/'valid_word_bank'; root.mkdir(exist_ok=True)
    for start in range(0,len(rows),32):
        chosen=rows[start:start+32]; ids=[r['signal'] for r in chosen]
        path=root/(f'{start:06d}.npz'); commit=path.with_suffix('.json')
        if commit.exists():
            c=read(commit)
            require(c['protocol_sha256']==protocol and c['signals']==ids and sha(path)==c['sha256'],
                    'Committed valid word cache changed')
        else:
            lengths=[]; values=[]; masks=[]; entries=[]
            for r in chosen:
                x,k,e=one(source,r['signal'],fast,core)
                values.append(x); masks.append(k); lengths.append(len(x)); entries.append(e)
            arrays=dict(signals=np.asarray(ids),lengths=np.asarray(lengths,np.int32),
                features=np.concatenate(values),evidence_mask=np.concatenate(masks),protocol_sha256=np.asarray(protocol))
            save_arrays(path,**arrays)
            c=dict(protocol_sha256=protocol,signals=ids,sha256=sha(path),entries=entries)
            freeze(commit,c)
        parts.append(dict(file=path.name,sha256=c['sha256'],signals=ids))
        emit('POOLED_VALID_WORD_PROGRESS',dict(done=start+len(ids),total=len(rows),seconds=round(time.perf_counter()-tick,1),
            missing_Audio2_encoded_this_process=source.encoded_missing))
    source.assert_matchers_unchanged(); source.release_encoder()
    summary=dict(protocol_sha256=protocol,signals_sha256=digest([r['signal'] for r in rows]),
                 parts=parts,records=len(rows),new_full_frame_cache_written=False)
    freeze(root/'complete.json',summary)
    return summary


def chunks(out, complete, protocol):
    for part in complete['parts']:
        path=out/'valid_word_bank'/part['file']
        require(path.resolve().parent==(out/'valid_word_bank').resolve() and sha(path)==part['sha256'], 'Invalid valid bank chunk')
        with np.load(path,allow_pickle=False) as z:
            require(z['signals'].tolist()==part['signals'] and z['protocol_sha256'].item()==protocol, 'Valid bank identity differs')
            lengths=z['lengths'].copy(); values=z['features'].copy(); mask=z['evidence_mask'].copy()
        require(lengths.shape==(len(part['signals']),) and np.issubdtype(lengths.dtype,np.integer)
                and (lengths>0).all() and (lengths<=100).all() and values.shape==(int(lengths.sum()),261)
                and values.dtype==np.float16 and mask.shape==(len(values),) and mask.dtype==bool
                and np.isfinite(values).all(), 'Malformed valid bank arrays')
        width=int(lengths.max()); v=np.zeros((len(lengths),width,261),np.float32)
        k=np.zeros((len(lengths),width),bool); w=np.arange(width)[None,:]<lengths[:,None]; offset=0
        for i,n in enumerate(lengths): v[i,:n]=values[offset:offset+n]; k[i,:n]=mask[offset:offset+n]; offset+=n
        yield part['signals'],v,k,w
