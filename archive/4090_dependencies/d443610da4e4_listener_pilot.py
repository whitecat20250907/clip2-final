"""Bounded A-prime pilot: decoder LoRA learns released listener transcriptions.

Uses one pre-existing nested TRAIN fold, fixed two epochs, and held-out reference
text for scoring. Never trains/selects on official validation. No raw audio/cache
copy and no foundation-model download. Adapter parameters alone are saved.
"""
from __future__ import annotations
import argparse
import hashlib
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from .common import digest, environment, owned, save_json, seed_all
from .labels import normalize
from .report import partial_correlation
from .train import metrics
from .response_diagnostics import listener_target
from research.personal_clip2.alignment import align_words

K5_GENERATION_KWARGS={'language':'english','task':'transcribe','return_timestamps':False,
    'do_sample':True,'temperature':.7,'top_k':0,'top_p':1.,'num_beams':1,
    'num_return_sequences':5,'max_new_tokens':128,'begin_suppress_tokens':[],
    'return_dict_in_generate':False}


def behavior_fingerprint(model_path):
    model_path=Path(model_path)
    names=('config.json','generation_config.json','preprocessor_config.json','tokenizer.json',
        'tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','added_tokens.json','normalizer.json')
    code=Path(__file__).resolve()
    return {'model_files':{name:digest(model_path/name) if (model_path/name).is_file() else None for name in names},
        'response_diagnostics_sha256':digest(code.with_name('response_diagnostics.py')),
        'normalization_sha256':digest(code.with_name('labels.py')),
        'alignment_sha256':digest(code.parents[1]/'research/personal_clip2/alignment.py'),
        'generation_overrides':K5_GENERATION_KWARGS}


class LoRALinear(nn.Module):
    def __init__(self,base,rank=16,alpha=32,dropout=.05):
        super().__init__(); self.base=base; self.scale=alpha/rank
        self.base.requires_grad_(False)
        self.lora_a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5)); self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        direct=self.base(x)
        residual=nn.functional.linear(nn.functional.linear(self.dropout(x).float(),self.lora_a),self.lora_b)
        return direct+residual.to(direct.dtype)*self.scale


def attach_adapters(model):
    model.requires_grad_(False); names=[]
    for name,module in list(model.named_modules()):
        if '.decoder.layers.' in name and name.endswith(('.q_proj','.v_proj')) and isinstance(module,nn.Linear):
            parent,child=name.rsplit('.',1)
            setattr(model.get_submodule(parent),child,LoRALinear(module)); names.append(name)
    if not names: raise ValueError('No decoder q/v projections found; architecture mismatch')
    return names


def word_spans(text):
    import re
    return [(m.start(),m.end()) for m in re.finditer(r'\S+',text)]


def offsets_to_words(text,offsets):
    spans=word_spans(text); result=[]
    for left,right in offsets:
        touched=[i for i,(a,b) in enumerate(spans) if max(a,left)<min(b,right)]
        if len(touched)>1: raise ValueError('Tokenizer token spans multiple reference words')
        result.append(touched[0] if touched else -1)
    if set(i for i in result if i>=0)!=set(range(len(spans))):
        raise ValueError('Tokenizer offsets do not cover every reference word')
    return result


def fixed_subset(records,available,max_prompts,seed):
    subset=[r for r in records if r['signal'] in available]
    prompts=sorted({r['prompt'] for r in subset})
    rng=np.random.default_rng(seed); rng.shuffle(prompts)
    keep=set(prompts[:max_prompts])
    return sorted([r for r in subset if r['prompt'] in keep],key=lambda r:r['signal'])


class AudioReader:
    def __init__(self,root,processor,cache_items=32):
        self.root=Path(root); self.processor=processor; self.cache=OrderedDict(); self.limit=cache_items
        self.attention_masks={}

    def __call__(self,sid):
        if sid in self.cache:
            value=self.cache.pop(sid); self.cache[sid]=value; return value
        wave,rate=sf.read(self.root/f'{sid}.flac',dtype='float32',always_2d=True)
        wave=wave[:,0]  # Fixed left ear, not chosen using human correctness.
        if rate!=16000:
            divisor=math.gcd(int(rate),16000)
            wave=resample_poly(wave,16000//divisor,int(rate)//divisor).astype(np.float32)
        if len(wave)>30*16000:
            raise ValueError('Audio longer than Whisper window; do not silently truncate')
        encoded=self.processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        value=encoded.input_features
        self.attention_masks[sid]=encoded.attention_mask
        self.cache[sid]=value
        while len(self.cache)>self.limit:
            evicted,_=self.cache.popitem(last=False); self.attention_masks.pop(evicted)
        return value

    def attention_mask(self,sid):
        self(sid)
        return self.attention_masks[sid]


def tokenize(tokenizer,text,device,*,offsets=False):
    encoded=tokenizer(text,add_special_tokens=True,return_offsets_mapping=offsets,
                      return_special_tokens_mask=True)
    ids=encoded['input_ids']
    if len(ids)>448: raise ValueError('Transcript exceeds decoder context')
    decoder=torch.tensor([ids[:-1]],device=device)
    target=torch.tensor([ids[1:]],device=device)
    loss_target=target.clone()
    # Language/task prefixes are provided, not learned targets. EOS remains a target.
    for i,(token,special) in enumerate(zip(ids[1:],encoded['special_tokens_mask'][1:])):
        if special and token!=tokenizer.eos_token_id: loss_target[0,i]=-100
    mapping=offsets_to_words(text,encoded['offset_mapping'][1:]) if offsets else None
    return decoder,target,loss_target,mapping


def evaluate_reference(model,processor,reader,records,device,data_index,data,deadline):
    model.eval(); word_predictions=[]; sentence=[]; covered_labels=[]; masked_prob=[]
    with torch.no_grad():
        for k,record in enumerate(records):
            if time.monotonic()>deadline: raise TimeoutError('A-prime pilot wall-clock cap reached')
            decoder,target,_,mapping=tokenize(processor.tokenizer,record['prompt'],device,offsets=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(input_features=reader(record['signal']).to(device),attention_mask=reader.attention_mask(record['signal']).to(device),decoder_input_ids=decoder,
                             use_cache=False).logits
            logprob=logits.float().log_softmax(-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)[0]
            token_prob=logprob.exp().cpu().numpy()
            probability=np.asarray([float(np.mean([token_prob[i] for i,j in enumerate(mapping) if j==word]))
                                   for word in range(int(record['n_words']))],np.float32)
            word_predictions.append(probability); sentence.append(float(probability.mean()))
            index=data_index[record['signal']]; mask=data['label_mask'][index,:len(probability)]
            covered_labels.extend(data['word_targets'][index,:len(probability)][mask].tolist())
            masked_prob.extend(probability[mask].tolist())
            if k%100==0: print(json.dumps({'stage':'reference_scoring','done':k,'total':len(records)}),flush=True)
    yy=np.asarray(covered_labels); pp=np.asarray(masked_prob)
    return word_predictions,np.asarray(sentence),{
        'word_auc':float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,
        'word_brier':float(np.mean((pp-yy)**2)),'labeled_words':len(yy)}


def candidate_word_frequencies(prompt, hypotheses):
    if len(hypotheses)!=5: raise ValueError('Require exactly five sampled hypotheses, including repeats')
    labels=[align_words(prompt,normalize(text) or '')['labels'] for text in hypotheses]
    result=np.asarray(labels,np.float32)
    if result.shape!=(5,len(prompt.split())): raise ValueError('Candidate/reference word mismatch')
    return result.mean(0)


def evaluate_k5(model,processor,reader,records,device,data_index,data,deadline):
    """Fixed sampling setup; same per-signal RNG for frozen/adapted comparison."""
    model.eval(); predictions=[]; sentences=[]; targets=[]; probabilities=[]; transcripts=[]; capped=0
    with torch.no_grad():
        for k,record in enumerate(records):
            if time.monotonic()>deadline: raise TimeoutError('A-prime pilot wall-clock cap reached')
            sample_seed=int.from_bytes(hashlib.sha256(('K5/2042/'+record['signal']).encode()).digest()[:4],'little')
            with torch.random.fork_rng(devices=[device.index if device.index is not None else 0]):
                torch.manual_seed(sample_seed)
                with torch.autocast('cuda',dtype=torch.float16):
                    sequences=model.generate(input_features=reader(record['signal']).to(device),
                        attention_mask=reader.attention_mask(record['signal']).to(device),
                        **K5_GENERATION_KWARGS)
            if sequences.shape[0]!=5: raise ValueError('Whisper did not return five samples')
            # Do not suppress initial EOS: a listener-imitation model must be able
            # to emit an empty response. The same setting is used in both arms.
            capped+=int((~(sequences==processor.tokenizer.eos_token_id).any(dim=1)).sum())
            text=processor.batch_decode(sequences,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            p=candidate_word_frequencies(record['prompt'],text)
            predictions.append(p); sentences.append(float(p.mean())); transcripts.append(text)
            index=data_index[record['signal']]; mask=data['label_mask'][index,:len(p)]
            targets.extend(data['word_targets'][index,:len(p)][mask].tolist()); probabilities.extend(p[mask].tolist())
            if k%100==0: print(json.dumps({'stage':'k5_scoring','done':k,'total':len(records)}),flush=True)
    yy=np.asarray(targets); pp=np.asarray(probabilities)
    return predictions,np.asarray(sentences),{'word_auc':float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,
        'word_brier':float(np.mean((pp-yy)**2)),'labeled_words':len(yy),
        'sequences_without_terminal_eos':capped,'sampled_sequences':5*len(records)},transcripts


def run(args):
    out=owned(args.output,args.work_root)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    # Redirect potential library state to our root and disallow network downloads.
    os.environ['HF_HOME']=str(owned(Path(args.work_root)/'hf_runtime',args.work_root))
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(2); seed_all(2042)
    if not torch.cuda.is_available(): raise RuntimeError('Use the authorized 4090 GPU')
    from transformers import AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
    device=torch.device('cuda:0'); started=time.monotonic(); deadline=started+min(args.max_hours,24)*3600
    root=Path(args.source_root); records=json.loads((root/'data/clip2/metadata/train_metadata.json').read_text())
    responses=json.loads((root/'data/clip2/metadata/train_responses.json').read_text())
    data=dict(np.load(Path(args.inputs)/'train.npz',allow_pickle=False))
    input_audit=json.loads((Path(args.inputs)/'preparation_audit.json').read_text())
    if digest(Path(args.inputs)/'train.npz')!=input_audit['splits']['train']['sha256']:
        raise ValueError('Prepared TRAIN digest mismatch')
    if input_audit['word_fields'][5]!='v3.left.match_prob' or input_audit['word_fields'][14]!='v3.right.match_prob':
        raise ValueError('Unexpected v3 feature layout')
    if 'token_identity_audit' not in input_audit: raise ValueError('Missing token identity audit')
    data_index={str(s):i for i,s in enumerate(data['signal_ids'])}
    folds=json.loads(Path(args.fold_roles).read_text())['roles']
    fit=fixed_subset(records,set(folds['train']),args.fit_prompts,2042)
    test=fixed_subset(records,set(folds['valid']),args.holdout_prompts,2043)
    if set(r['prompt'] for r in fit)&set(r['prompt'] for r in test): raise ValueError('Prompt leakage')
    if not fit or not test or any(r['signal'] not in data_index for r in fit+test): raise ValueError('Pilot must stay inside canonical TRAIN')
    if Path(args.d6e_oof).resolve().parent!=Path(args.fold_roles).resolve().parent.parent:
        raise ValueError('D6e OOF predictions and fold roles must belong to the same nested experiment')
    oof=dict(np.load(args.d6e_oof,allow_pickle=False)); lookup={str(s):i for i,s in enumerate(oof['signal_ids'])}
    fold_prediction_path=Path(args.fold_roles).parent/'d6e_seed42/valid_predictions.npz'
    fold_prediction=dict(np.load(fold_prediction_path,allow_pickle=False))
    fold_lookup={str(s):i for i,s in enumerate(fold_prediction['signal_ids'])}
    if set(fold_lookup)!=set(folds['valid']) or len(fold_lookup)!=len(fold_prediction['signal_ids']):
        raise ValueError('Fold predictions do not cover the exact outer holdout')
    if len(lookup)!=len(oof['signal_ids']): raise ValueError('Duplicate OOF IDs')
    for record in test:
        sid=record['signal']
        if sid not in lookup or not np.isfinite(oof['d6e'][lookup[sid]]): raise ValueError('Missing finite D6e OOF prediction')
        if float(oof['target'][lookup[sid]])!=float(data['target'][data_index[sid]]): raise ValueError('OOF target mismatch')
        if float(oof['d6e'][lookup[sid]])!=float(fold_prediction['prediction'][fold_lookup[sid]]):
            raise ValueError('OOF value differs from this fold model prediction')
        if float(fold_prediction['target'][fold_lookup[sid]])!=float(data['target'][data_index[sid]]):
            raise ValueError('Fold target mismatch')
    model_path=Path(args.model)
    if not (model_path/'model.safetensors').is_file(): raise FileNotFoundError('Require existing local safetensors weights')
    processor=WhisperProcessor.from_pretrained(model_path,local_files_only=True)
    processor.tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=True,local_files_only=True)
    if not processor.tokenizer.is_fast: raise ValueError('Fast tokenizer offsets are required for auditable word alignment')
    processor.tokenizer.set_prefix_tokens(language='english',task='transcribe',predict_timestamps=False)
    model=WhisperForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
            use_safetensors=True,torch_dtype=torch.float16,attn_implementation='eager').to(device)
    model.requires_grad_(False)
    reader=AudioReader(root/'data/clip2/train/signals',processor)
    config={'stage':'A-prime bounded pilot','ear':'left','epochs':2,'fit_prompts_cap':args.fit_prompts,
        'holdout_prompts_cap':args.holdout_prompts,'fit_rows':len(fit),'holdout_rows':len(test),
        'batch_size':1,'gradient_accumulation':4,'learning_rate':2e-5,'rank':16,'alpha':32,
        'deadline_hours':min(args.max_hours,24),'seed':2042,'environment':environment(),
        'source_model_config_sha256':digest(model_path/'config.json'),
        'source_model_safetensors_sha256':digest(model_path/'model.safetensors'),
        'fold_roles_sha256':digest(args.fold_roles),'input_audit_sha256':digest(Path(args.inputs)/'preparation_audit.json'),
        'd6e_oof_sha256':digest(args.d6e_oof),'d6e_fold_prediction_sha256':digest(fold_prediction_path),
        'fit_signal_ids':[r['signal'] for r in fit],'holdout_signal_ids':[r['signal'] for r in test],
        'checkpoint_selection':'fixed epoch 2; holdout not used to select or tune',
        'word_probability':'arithmetic mean of teacher-forced token probabilities for each reference word',
        'v3_comparison':'existing better-ear zero-gated 1-best word probability; not same-model TF'}
    save_json(out/'protocol.json',config)
    frozen_words,frozen_sentence,frozen_metrics=evaluate_reference(model,processor,reader,test,device,data_index,data,deadline)
    adapted_names=attach_adapters(model)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda'); history=[]
    for epoch in (1,2):
        rng=np.random.default_rng(2042+epoch); order=rng.permutation(len(fit))
        model.train(); optimizer.zero_grad(set_to_none=True); total=0.; used=0
        for step,index in enumerate(order):
            if time.monotonic()>deadline: raise TimeoutError('A-prime pilot wall-clock cap reached')
            record=fit[index]; rr=responses.get(record['signal'])
            if not isinstance(rr,list) or len(rr)!=1: raise ValueError('Expected one released listener response')
            text=listener_target(rr[0])
            if text is None: raise ValueError('Missing listener transcription, not a zero target')
            if text=='xxx': text=''
            decoder,_,targets,_=tokenize(processor.tokenizer,text,device)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(input_features=reader(record['signal']).to(device),attention_mask=reader.attention_mask(record['signal']).to(device),decoder_input_ids=decoder,use_cache=False).logits
                loss=nn.functional.cross_entropy(logits.float().flatten(0,1),targets.flatten(),ignore_index=-100)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite listener-imitation loss')
            # Final partial accumulation group uses its actual size, not four.
            group_size=min(4,len(order)-(step//4)*4)
            scaler.scale(loss/group_size).backward(); total+=float(loss.detach()); used+=1
            if (step+1)%4==0 or step+1==len(order):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(parameters,1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step%100==0: print(json.dumps({'stage':'listener_lora','epoch':epoch,'step':step,'total':len(order),'mean_loss':total/used}),flush=True)
        history.append({'epoch':epoch,'mean_token_cross_entropy':total/used,'samples':used})
        save_json(out/'history.json',history)
    adapter_state={name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
    torch.save({'state':adapter_state,'modules':adapted_names,'protocol':config,'training_role':'TRAIN_fold_listener_pilot_not_official_submission'},out/'adapter.pt')
    word,sentence,after=evaluate_reference(model,processor,reader,test,device,data_index,data,deadline)
    indices=np.asarray([data_index[r['signal']] for r in test]); subset=data['words'][indices]
    # Original schema: v3-left match_prob at 5, v3-right match_prob at 14.
    v3=np.maximum(subset[:,:,5],subset[:,:,14]); mask=data['label_mask'][indices]
    yy=data['word_targets'][indices][mask]; pp=v3[mask]
    v3_auc=float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None
    base=np.asarray([oof['d6e'][lookup[r['signal']]] for r in test])
    target=data['target'][indices]
    controls=np.column_stack([base,np.eye(3)[data['severity'][indices]],data['n_words'][indices]])
    rho=partial_correlation(sentence,target-base,controls)
    frozen_rho=partial_correlation(frozen_sentence,target-base,controls)
    gain=after['word_auc']-v3_auc if after['word_auc'] is not None and v3_auc is not None else None
    same_gain=after['word_auc']-frozen_metrics['word_auc'] if after['word_auc'] is not None and frozen_metrics['word_auc'] is not None else None
    result={'frozen_medium':frozen_metrics,'adapted_medium':after,'original_v3_word_auc':v3_auc,
        'delta_word_auc_vs_v3':gain,'delta_word_auc_vs_same_medium':same_gain,
        'oof_partial_residual_correlation':rho,'auc_gain_threshold':.02,'partial_correlation_threshold':.08,
        'frozen_medium_partial_residual_correlation':frozen_rho,
        'delta_partial_correlation_vs_same_medium':rho-frozen_rho if rho is not None and frozen_rho is not None else None,
        'outline_threshold_passed':bool(gain is not None and rho is not None and gain>=.02 and rho>=.08),
        'lora_point_estimate_improves_same_medium':bool(same_gain is not None and same_gain>0 and rho is not None and frozen_rho is not None and rho>frozen_rho),
        'expansion_eligible':bool(gain is not None and rho is not None and gain>=.02 and rho>=.08 and same_gain is not None and same_gain>0 and frozen_rho is not None and rho>frozen_rho),
        'sentence_proxy_metrics':metrics(sentence,target),'official_rmse':None,
        'seconds':time.monotonic()-started,'adapter_sha256':digest(out/'adapter.pt'),
        'caveats':['fixed small prompt-disjoint TRAIN fold pilot, not full-population OOF',
                   'TF word probabilities are an exploratory sentence proxy, not a fitted D6e replacement',
                   'left-ear adapted medium versus better-ear v3 changes model family/context; same-medium control is also reported']}
    max_words=max(len(p) for p in word); wp=np.zeros((len(word),max_words),np.float32); fw=np.zeros_like(wp)
    for i,(a,b) in enumerate(zip(word,frozen_words)): wp[i,:len(a)]=a; fw[i,:len(b)]=b
    np.savez(out/'holdout_predictions.npz',signal_ids=np.asarray([r['signal'] for r in test]),
             prediction=sentence,frozen_prediction=frozen_sentence,word_prediction=wp,
             frozen_word_prediction=fw,target=target,d6e_oof_prediction=base)
    save_json(out/'results.json',result); print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ('source-root','inputs','model','fold-roles','d6e-oof','work-root','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--fit-prompts',type=int,default=600)
    p.add_argument('--holdout-prompts',type=int,default=300)
    p.add_argument('--max-hours',type=float,default=24)
    run(p.parse_args())


if __name__=='__main__': main()
