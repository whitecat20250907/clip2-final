"""Public-only word mapping; all own canonical records remain present."""
import re
import unicodedata
from pathlib import Path
import numpy as np
import torch
from utils import require,read,sha,digest,emit
import frozen_inputs


def words(text):
    text=unicodedata.normalize('NFKC',text).lower().replace('’',"'").replace('‘',"'")
    bad=re.findall(r'[^a-z\s\.,!?;:\-—–\x27"()…]',text)
    if bad:
        codes=sorted({f'U+{ord(c):04X}' for c in bad})
        raise ValueError('Unsupported scoring/source lyric characters: '+', '.join(codes))
    return re.findall(r"[a-z]+(?:'[a-z]+)*'?",text)


def source_sequence(text):
    """Opaque unsupported source spans never become partial/guessed words.

    Preserve a nonmatching sentinel at each rejected source position so it is
    not silently joined to neighboring tokens. Scoring words remain strict.
    """
    spans=list(re.finditer(r'\S+',text)); sequence=[]; owners=[]; rejected=[]
    for i,span in enumerate(spans):
        try: normalized=words(span.group())
        except ValueError:
            canonical=unicodedata.normalize('NFKC',span.group()).lower().replace('’',"'").replace('‘',"'")
            bad=re.findall(r'[^a-z\s\.,!?;:\-—–\x27"()…]',canonical)
            rejected.append(dict(whitespace_index=i,span_start=span.start(),span_end=span.end(),
                public_source_span=span.group(),unsupported_characters=sorted(set(bad)),
                unicode_codepoints=sorted({f'U+{ord(c):04X}' for c in bad})))
            normalized=[None]
        sequence.extend(normalized); owners.extend([i]*len(normalized))
    return spans,sequence,owners,rejected


def unique_lcs(source,target):
    """Accept only a match present in every optimal LCS alignment."""
    n,m=len(source),len(target); f=np.zeros((n+1,m+1),np.int32); b=np.zeros_like(f)
    for i in range(n):
        for j in range(m): f[i+1,j+1]=max(f[i,j+1],f[i+1,j],f[i,j]+int(source[i]==target[j]))
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1): b[i,j]=max(b[i+1,j],b[i,j+1],b[i+1,j+1]+int(source[i]==target[j]))
    optimum=int(f[n,m]); result={}
    for i in range(n):
        matches=[j for j in range(m) if source[i]==target[j] and f[i,j]+1+b[i+1,j+1]==optimum]
        if len(matches)!=1 or any(f[i,j]+b[i+1,j]==optimum for j in range(m+1)): continue
        j=matches[0]
        reverse=[k for k in range(n) if source[k]==target[j] and f[k,j]+1+b[k+1,j+1]==optimum]
        if len(reverse)==1 and not any(f[k,j]+b[k,j+1]==optimum for k in range(n+1)): result[i]=j
    return result


def word_vectors(encoded_text,scoring_text,offsets,embeddings):
    """Whitespace token owners are admitted only if one canonical word.

    BOS/EOS (zero length offsets), punctuation-only and ambiguous token spans
    do not enter word representations. Unmatched/truncated words remain masked.
    """
    scoring=words(scoring_text); spans,source,owners,_=source_sequence(encoded_text)
    mapping=unique_lcs(source,scoring); eligible={}
    for i,j in mapping.items():
        owner=owners[i]
        if owners.count(owner)==1: eligible[owner]=j
    out=np.zeros((len(scoring),embeddings.shape[1]),np.float32); counts=np.zeros(len(scoring),np.int64)
    require(len(offsets)==len(embeddings),'Embedding/token count mismatch')
    for (a,b),embedding in zip(offsets,embeddings):
        if b<=a: continue
        # Strip tokenizer-leading whitespace before assigning overlapping spans.
        while a<b and encoded_text[a].isspace(): a+=1
        while b>a and encoded_text[b-1].isspace(): b-=1
        hits=[i for i,s in enumerate(spans) if a<s.end() and b>s.start()]
        if len(hits)==1 and hits[0] in eligible and re.search(r'[a-zA-Z]',encoded_text[a:b]):
            j=eligible[hits[0]]; out[j]+=embedding; counts[j]+=1
    out/=np.maximum(counts,1)[:,None]
    return out,counts>0,scoring


def shuffle_indices(tokens,mask):
    """Within-record word reassignment preserves all lengths/masks/text content."""
    positions=np.flatnonzero(mask); result=np.arange(len(tokens))
    if len(positions)<2: return result,0
    # Choose rotation with maximum changed lexical identities, fixed public inputs.
    choices=[]
    for shift in range(1,len(positions)):
        donor=np.roll(positions,shift)
        changed=sum(tokens[a]!=tokens[b] for a,b in zip(positions,donor))
        choices.append((changed,-shift,donor))
    changed,_,donor=max(choices,key=lambda v:(v[0],v[1])); result[positions]=donor
    return result,int(changed)


def cache_encoded_texts(rows,wanted):
    """Reproduce reviewed extractor's setdefault in ORIGINAL metadata order.

    A prompt key can have several original lyrics. The saved cache has ONE
    representation for that key: the first occurrence, not the latest row or
    whichever alternative happens to match token length. Keep every conflict
    in a public-only audit; downstream tokenizer checks remain mandatory.
    """
    encoding={}; variants={}; occurrences={}; first_signals={}
    for row in rows:
        key=str(row['prompt'])
        if key not in wanted: continue
        raw=row.get('original_prompt') or row['prompt']
        require(isinstance(raw,str),'Invalid public original prompt')
        if key not in encoding:
            encoding[key]=raw; first_signals[key]=str(row.get('signal',''))
        variants.setdefault(key,set()).add(raw)
        occurrences[key]=occurrences.get(key,0)+1
    require(set(encoding)==set(wanted),'Reviewed original lyric inputs missing')
    conflicts={p:sorted(v) for p,v in variants.items() if len(v)>1}
    return encoding,dict(rule='First occurrence in pinned metadata order, reproduces reviewed extractor key2text.setdefault',
        keys=len(encoding),conflicting_original_lyrics_keys=len(conflicts),
        conflicting_key_record_count=sum(occurrences[p] for p in conflicts),
        encoded_public_text_sha256=digest(encoding),conflicting_variants_sha256=digest(conflicts),
        first_source_signal_sha256=digest(first_signals),all_conflicting_variants=conflicts,
        alternatives_selected_by_tokenizer_or_scores=False,rows_dropped=0,
        generator_current_cache_hash_verified=False)


class WordCache:
    def __init__(self, original,bank,logits,text,masks,shuffled,identity):
        self.original=original; self.bank=bank; self.logits=logits
        self.text=text; self.masks=masks; self.shuffled=shuffled
        self.identity=identity; self.ram=original.ram

    def batch(self,selected,text_prompts,base,device):
        ii=np.asarray([self.original.index[s] for s in selected]); lengths=self.original.lengths[ii]
        t=int(lengths.max()); audio=np.zeros((len(ii),t,1280),np.float32)
        am=np.arange(t)[None,:]<lengths[:,None]
        v,k,w=self.bank.batch(selected,0)
        v,k,w=(x.to(device) for x in (v,k,w)); width=w.shape[1]
        text=np.zeros((len(ii),width,768),np.float32); mask=np.zeros((len(ii),width),bool)
        for j,(sid,i) in enumerate(zip(selected,ii)):
            n=int(lengths[j]); o=int(self.original.offsets[i]); audio[j,:n]=self.original.frames[o:o+n]
            tx=self.text[sid]; require(len(tx)==int(w[j].sum()),'Scoring word count differs')
            text[j,:len(tx)]=tx[self.shuffled[sid]] if text_prompts is self.shuffled else tx
            mask[j,:len(tx)]=self.masks[sid]
        require(np.isfinite(audio).all(),'Nonfinite audio input')
        return (torch.as_tensor(audio,device=device),torch.as_tensor(am,device=device),
            torch.as_tensor(text,device=device),torch.as_tensor(mask,device=device),v,k,w,
            torch.as_tensor([self.logits[s] for s in selected],device=device),
            torch.as_tensor([base[s] for s in selected],device=device))

    def verify_stat(self): self.original.verify_stat()


def prepare(peer,out):
    ctx=frozen_inputs.prepare(peer,out); f=ctx['frozen']; retained=ctx.pop('retained')
    # This reviewed metadata identifies text content under scoring-prompt keys.
    evidence=read(Path(__file__).parent/'PEER_EVIDENCE.json')
    path=Path('/home/lisiyi/icassp2data/clip2/metadata/train_metadata.json')
    reviewed=evidence['replay']['protocol']['source_sha256'][str(path)]
    require(sha(path)==reviewed,'Reviewed text-generation metadata changed')
    wanted=set(ctx['prompts'].values())
    encoding,key_audit=cache_encoded_texts(read(path),wanted)
    # Audit all variants without printing potentially large public lyric arrays.
    from utils import freeze
    freeze(out/'cache_key_audit.json',key_audit)
    emit('POOLED_WORD_CACHE_KEY_AUDIT',{k:v for k,v in key_audit.items() if k!='all_conflicting_variants'})
    text_path=peer/'output/text_emb_roberta_wid_all.npz'
    require(sha(text_path)==ctx['cache'].identity['text_sha256'],'Text cache changed before mapping')
    with np.load(text_path,allow_pickle=True) as z:
        require({'emb','offsets','lengths','prompts','word_ids','model'}<=set(z.files),'Text cache lacks mapping/model identity')
        model_id=str(z['model'].item()); require('roberta' in model_id.lower(),'Only reviewed RoBERTa cache supported')
        emb=z['emb'].copy(); off=z['offsets'].copy(); lens=z['lengths'].copy()
        keys=[str(s) for s in z['prompts']]; stored_ids=z['word_ids'].copy()
    # Offline only. Never silently download/re-extract or substitute a tokenizer.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_id,local_files_only=True,use_fast=True)
    require(tokenizer.is_fast,'Offsets require local fast tokenizer')
    tokenize_hash=digest(tokenizer.backend_tokenizer.to_str())
    rejected={p:source_sequence(encoding[p])[-1] for p in sorted(wanted)}
    rejected={p:v for p,v in rejected.items() if v}
    from collections import Counter
    codes=Counter(c for items in rejected.values() for item in items for c in item['unicode_codepoints'])
    normalization_audit=dict(policy='Strict scoring normalization unchanged. Unsupported original-text whitespace spans become opaque unmatched positions; their text injection masked, no guessed transliteration or punctuation stripping.',
        rejected_prompt_keys=len(rejected),rejected_source_spans=sum(map(len,rejected.values())),
        unicode_codepoint_span_counts=dict(codes),public_rejected_spans=rejected,
        original_text_rewritten=False,scoring_words_removed=False,records_removed=0,
        TRAIN_or_DEV_score_values_used=False,coverage_threshold_changed=False)
    freeze(out/'source_normalization_audit.json',normalization_audit)
    emit('POOLED_WORD_SOURCE_NORMALIZATION_AUDIT',{k:v for k,v in normalization_audit.items() if k!='public_rejected_spans'})
    lookup={p:i for i,p in enumerate(keys)}; vectors={}; masks={}; tokens={}; truncated=0
    for p in sorted(wanted):
        i=lookup[p]; o=int(off[i]); n=int(lens[i]); raw=encoding[p]
        full=tokenizer(raw,add_special_tokens=True,return_offsets_mapping=True,truncation=False)
        # Reviewed extractor caps64; a longer current cache is accepted only if
        # tokenization length exactly agrees with its saved per-prompt length.
        encoded=tokenizer(raw,add_special_tokens=True,return_offsets_mapping=True,truncation=True,max_length=n)
        offsets=encoded['offset_mapping']; require(len(offsets)==n,'Text cache/tokenizer length disagrees: '+p)
        spans=list(re.finditer(r'\S+',raw)); expected=[]
        for a,_ in offsets:
            choices=[j for j,s in enumerate(spans) if a>=s.start()]; expected.append(max(choices,default=0))
        require(np.array_equal(np.asarray(expected),stored_ids[o:o+n]),'Stored word IDs do not match reviewed original lyric recipe: '+p)
        vectors[p],masks[p],tokens[p]=word_vectors(raw,p,offsets,emb[o:o+n])
        truncated+=int(len(full['input_ids'])>n)
    # Reconstruct DIRECT saved D6e logits; pooled anchor replay already passed.
    baseline,residual=retained['bundle'].models[43]; bank=retained['bank']; logits={}
    infer=retained['infer']; raw=retained['raw']; sides=infer.normalized_side(raw,retained['bundle'].norm)
    with torch.no_grad():
        for start in range(0,len(f.ids),128):
            selected=f.ids[start:start+128]; ix=[f.index[s] for s in selected]
            x=torch.from_numpy(raw['pooled'][ix]); s=torch.from_numpy(sides[ix])
            z=baseline.head[:-1](torch.cat((baseline.projection(x),s),-1)).squeeze(-1)
            logits.update(zip(selected,z.tolist()))
    text={}; valid={}; shuffled={}; stats={}
    for partition,ids in [('TRAIN',f.partition['train_ids']),('DEV',f.partition['dev_ids'])]:
        total=mapped=changed=eligible=rows_changed=rows_empty=0
        for sid in ids:
            p=ctx['prompts'][sid]; tx=vectors[p]; mask=masks[p]
            require(len(tx)==int(bank.lengths[bank.index[sid]]),'Canonical word count/cache differs: '+sid)
            support=bank.support[0,bank.offsets[bank.index[sid]]:bank.offsets[bank.index[sid]+1]].numpy()
            perm,n=shuffle_indices(tokens[p],mask & support)
            text[sid]=tx; valid[sid]=mask; shuffled[sid]=perm
            total+=len(mask); mapped+=int(mask.sum()); eligible+=int((mask & support).sum()); changed+=n
            rows_changed+=int(n>0); rows_empty+=int(not (mask & support).any())
        stats[partition]=dict(records=len(ids),scoring_words=total,mapped_text_words=mapped,eligible_words=eligible,
            changed_lexical_assignments=changed,records_with_changed_assignment=rows_changed,
            records_without_eligible_word=rows_empty,rows_dropped=0)
    require(stats['TRAIN']['mapped_text_words']/stats['TRAIN']['scoring_words']>=.90,'Mapping coverage below90%; do not train')
    require(stats['TRAIN']['records_with_changed_assignment']/16204>=.80,'Shuffle changes fewer than80% TRAIN rows; do not train')
    identity=dict(metadata_sha256=reviewed,text_sha256=sha(text_path),tokenizer_id=model_id,
        cache_key_audit_sha256=sha(out/'cache_key_audit.json'),
        cache_key_rule=key_audit['rule'],conflicting_original_lyrics_keys=key_audit['conflicting_original_lyrics_keys'],
        source_normalization_audit_sha256=sha(out/'source_normalization_audit.json'),
        unsupported_source_span_policy=normalization_audit['policy'],
        rejected_source_prompt_keys=normalization_audit['rejected_prompt_keys'],
        offline_tokenizer_sha256=tokenize_hash,truncated_prompt_count=truncated,partitions=stats,
        vector_content_hash=digest({p:__import__('hashlib').sha256(v.tobytes()).hexdigest() for p,v in vectors.items()}),
        mapping='Original whitespace tokens to mandatory normalized scoring LCS; specials excluded using offsets; multiword/ambiguous/missing/truncated positions masked',
        cached_embedding_generator_hash_verified=False,
        limitation='Saved original-text metadata, token counts and word IDs agree; this does not independently reproduce RoBERTa embedding values.')
    ctx['saved_word_state']={k:v.detach().clone() for k,v in residual.state_dict().items()}
    ctx['cache']=WordCache(ctx['cache'],bank,logits,text,valid,shuffled,identity)
    ctx['wrong']=shuffled; ctx['identity']['word_mapping']=identity
    ctx['identity']['wrong_mapping_audit']=stats
    emit('POOLED_WORD_MAPPING_READY',identity)
    return ctx
