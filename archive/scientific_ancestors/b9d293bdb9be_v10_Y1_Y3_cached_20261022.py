"""BEEP machine evidence and cross-fitted lexical effects, no audio rerun."""
from pathlib import Path
import os,sys,collections,hashlib,json
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'v10_final15h_site')]
import v10_Y1_beep_replay_20261022 as B
from v10_Y1_replay_ties_20261022 import replay
from v10_Y_tiny_calibration_20261022 import fit
import v10_new_information_20261022 as G
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
T=G.T;D=T.V/'final15h_20261022';ER=T.V/'evidence_relations_20261022/ER01'

def load():
    report=T.read(D/'Y1/rapid_beep_replay_complete.json');assert report['below_half_percent'];tr=G.P.F.arr(ER/'train.npz');va=G.P.F.arr(ER/'valid.npz');ids=set(map(str,np.concatenate([tr['signal_ids'],va['signal_ids']])));records={}
    with (T.V/'whole_human_mishearing_20261022/sequence_inputs.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['signal_id'] in ids:records[r['signal_id']]=r
    assert len(records)==len(ids);return tr,va,records

def y1(tr,va,records):
    lex=B.dictionary();rules=T.read(G.D/'contraction_audit.json')['expansion'];align=G.alignment();features=[];counts=collections.Counter()
    for z in [tr,va]:
        x=[]
        for sid in z['signal_ids']:
            r=records[str(sid)];ref=r['reference'];row=[]
            for family in ['MF','MOSS']:
                src={s['stream']:s for s in r['sources'] if s['family']==family}
                for stream in ['A1L','A1R','A2L','A2R']:
                    if stream not in src:
                        row.append(0.);counts[family+'_absent_native_stream_neutral']+=1;continue
                    raw=src[stream]['words'];h,_,_=G.expand(raw,rules);rr,ri,_=G.expand(ref,rules);labels,mapping=replay(rr,h,lex,'rapid_all');old=align.align_words(' '.join(rr),' '.join(h))['labels'];delta=np.mean(labels)-np.mean(old);row.append(delta);counts[family+'_changed_streams']+=int(delta!=0)
            sev=int(r['severity']);row=np.asarray(row);x.append(np.concatenate([row,row[:4]*(sev==1),row[:4]*(sev==2)]))
        features.append(np.asarray(x,np.float32))
    fit('Y1',*features,[f'BEEP_vs_literal_{f}_{s}' for f in ['MF','MOSS'] for s in ['A1L','A1R','A2L','A2R']]+[f'MF_delta_mild_{s}' for s in range(4)]+[f'MF_delta_moderate_{s}' for s in range(4)],dict(replay=T.read(D/'Y1/rapid_beep_replay_complete.json'),counts=dict(counts),scope='Partial判词口径migration: add BEEP machine match differences around unchanged winning head, not full upstream retraining',no_confidence_duplication=True,unknown_words_literal_only=True))

def y3(tr,va,records):
    human=T.read(D/'Y1/rapid_beep_word_labels.json');blocks=[];group=[];base=[];words=[];response=[];row_index=[];trainrows=[];align=G.alignment()
    for i,sid in enumerate(tr['signal_ids']):
        r=records[str(sid)];h=human[str(sid)]
        if not h['hits_match']:continue
        # Verify against the existing all-optimal literal audit where literal
        # and BEEP labels agree. Homophone-specific ambiguity is retained as
        # a stated limitation, never relabeled using the official count.
        family=[]
        for fam in ['MF','MOSS']:
            src={s['stream']:s for s in r['sources'] if s['family']==fam};family.extend([np.asarray(align.align_words(' '.join(r['reference']),' '.join(src[s]['words']))['labels']) if s in src else np.zeros(len(r['reference'])) for s in ['A1L','A1R','A2L','A2R']])
        feature=np.stack(family,1);sev=r['severity'];feature=np.column_stack([np.ones(len(feature)),feature,np.full(len(feature),sev==1),np.full(len(feature),sev==2)]);trainrows.append((i,feature,r['reference'],h['labels'],int(hashlib.sha256(' '.join(r['reference']).encode()).hexdigest()[:8],16)%5))
    def fit_word(rows):
        x=np.concatenate([q[1] for q in rows]);y=np.concatenate([q[3] for q in rows]);n=len(y)
        def fun(w):
            p=expit(x@w);return float(np.mean(np.logaddexp(0,x@w)-y*(x@w))+.002*(w[1:]@w[1:])),x.T@(p-y)/n+.004*np.r_[0,w[1:]]
        q=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options=dict(maxiter=250));assert q.success;return q.x
    def table(rows,w):
        total=collections.defaultdict(lambda:[0.,0])
        for i,x,ws,ys,f in rows:
            err=np.asarray(ys)-expit(x@w)
            for word,e in zip(ws,err):total[word][0]+=float(e);total[word][1]+=1
        return {k:dict(effect=float(np.clip(s/(n+20),-.25,.25)) if n>=5 else 0.,count=n) for k,(s,n) in total.items()}
    x=np.zeros((len(tr['signal_ids']),1),np.float32);coverage=[]
    for fold in range(5):
        rows=[q for q in trainrows if q[-1]!=fold];w=fit_word(rows);effects=table(rows,w)
        for i,xx,ws,ys,f in trainrows:
            if f==fold:x[i,0]=np.mean([effects.get(word,dict(effect=0))['effect'] for word in ws])
        coverage.append(dict(fold=fold,training_phrases=len(rows),word_types=len(effects)))
    w=fit_word(trainrows);effects=table(trainrows,w);v=np.asarray([[np.mean([effects.get(word,dict(effect=0))['effect'] for word in records[str(s)]['reference']])] for s in va['signal_ids']],np.float32)
    # Rows with irreproducible word labels stay in sentence training with the
    # neutral lexical effect. No official sentence target is removed/changed.
    T.save(D/'Y3/lexical_table.json',dict(UTC=T.utc(),word_base_coefficients=w.tolist(),effects=effects,folds=coverage,minimum_count=5,prior_count=20,word_base_parameters=len(w)))
    fit('Y3',x,v,['crossfit_lexical_human_minus_machine_effect'],dict(lexical_types=len(effects),table_learned_parameters='One empirical-Bayes effect per supported word; not counted as one parameter model',word_base_parameters=len(w),calibration_parameters=1,prompt_groups_excluded_from_own_effect=True,unmatched_word_rows_neutral=True,label_count_not_exact_alignment_note=True))

if __name__=='__main__':
    tr,va,records=load()
    for fun in [y1,y3]:
        (D/('Y1' if fun==y1 else 'Y3')).mkdir(exist_ok=True)
        try:fun(tr,va,records)
        except Exception:
            import traceback;traceback.print_exc()
