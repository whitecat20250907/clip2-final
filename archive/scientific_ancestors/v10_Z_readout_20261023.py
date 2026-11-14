"""Fit tiny fixed corrections and submit each technically valid Z candidate once.

The TRAIN parent is a disclosed conditional surrogate inherited from ER01,
not a claim of complete joint nested OOF for the recursively updated winner.
"""
from pathlib import Path
import os,sys,json,csv,time,hashlib,traceback,collections
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'v10_final15h_site')]
import v10_new_information_20261022 as G
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
T=G.T;D=T.V/'second_round_20261023';ER=T.V/'evidence_relations_20261022/ER01';F=G.P.H.F

def parents():
    tr=G.P.F.arr(ER/'train.npz');va=G.P.F.arr(ER/'valid.npz');ref=D/'own_reference.json'
    if not ref.exists():T.save(ref,T.read(T.V/'current_best_own_valid.json'))
    baseline=T.read(ref);ids,prob=G.F.csvpred(baseline['csv']);assert np.array_equal(ids,va['signal_ids']);va['parent']=prob
    return tr,va,baseline

def candidate(tid,p,model,extra):
    tr,va,parent=parents();out=D/tid;out.mkdir(exist_ok=True);c=out/'candidate';c.mkdir(exist_ok=True)
    T.save(out/'model.json',model)
    assert p.shape==va['parent'].shape and np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=1
    with (c/'valid_predictions.csv').open('w',newline='') as f:csv.writer(f).writerows(zip(va['signal_ids'],[format(float(v)*100,'.10f') for v in p]))
    report=dict(UTC=T.utc(),task=tid,csv_SHA=T.sha(c/'valid_predictions.csv'),source_model_SHA=T.sha(out/'model.json'),frozen_parent=dict(submission_id=parent['submission_id'],RMSE=parent['RMSE'],csv=parent['csv'],csv_SHA=parent['csv_SHA']),TRAIN=3072,VALID=2117,TRAIN_parent='Inherited ER01 conditional surrogate, not exact current winner nested OOF; official-only adoption',TRAIN_score_submission_gate=False,VALID_labels=False,EVAL=False,extra=extra)
    T.save(c/'report.json',report);T.save(c/'verification.json',dict(status='verified',csv_sha256=report['csv_SHA'],report_sha256=T.sha(c/'report.json')))
    cid='v10_'+tid.replace('+','_')+'_fixed_second_round_20261023';reg=T.read(D/'official_registry.json') if (D/'official_registry.json').exists() else dict(candidates=[])
    if not any(r['id']==cid for r in reg['candidates']):
        spec=F.submit_candidates([F.candidate(c,cid,cid,'Fixed tiny correction; no TRAIN score gate; '+tid)],out/'official');reg['candidates'].append(dict(id=cid,task=tid,candidate=str(c),manifest=str(out/'official/manifest.json')));T.save(D/'official_registry.json',reg)
    print('Z_CANDIDATE_READY',tid,model['parameters'],flush=True)

def fit_sentence(tid,x,v,fields,extra):
    if (D/tid/'candidate/verification.json').exists():return
    tr,va,_=parents();assert x.shape==(len(tr['signal_ids']),len(fields)) and v.shape==(len(va['signal_ids']),len(fields)) and len(fields)<=20 and np.isfinite(x).all() and np.isfinite(v).all()
    mu=x.mean(0);sd=np.maximum(x.std(0),.01);a=np.clip((x-mu)/sd,-5,5);b=np.clip((v-mu)/sd,-5,5);l=logit(np.clip(tr['parent'],1e-8,1-1e-8));y=tr['target'];lam=.002
    def fun(w):
        u=a@w;s=np.tanh(u);q=expit(l+.2*s);e=q-y
        return float(np.mean(e*e)+lam*(w@w)),a.T@(2*e*q*(1-q)*.2*(1-s*s))/len(y)+2*lam*w
    fit=minimize(fun,np.zeros(len(fields)),jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-13,gtol=1e-9));assert fit.success,fit.message
    base=va['parent'];predict=lambda bp,xx:expit(logit(np.clip(bp,1e-8,1-1e-8))+.2*np.tanh(xx@fit.x));p=predict(base,b);zero=expit(logit(np.clip(base,1e-8,1-1e-8)));reverse=predict(base[::-1],b[::-1])[::-1];assert np.max(abs(zero-base))<1.1e-8 and np.max(abs(p-reverse))<1e-12 and np.max(abs(p-base))<.050001
    out=D/tid;out.mkdir(exist_ok=True);np.savez_compressed(out/'features.npz',train_signal_ids=tr['signal_ids'],valid_signal_ids=va['signal_ids'],train=x,valid=v)
    model=dict(parameters=len(fields),fields=fields,mean=mu.tolist(),std=sd.tolist(),weights=fit.x.tolist(),logit_bound=.2,regularization=lam,optimizer='fixed L-BFGS-B500')
    extra.update(feature_SHA=T.sha(out/'features.npz'),zero_replay_max_prob=float(np.max(abs(zero-base))),reverse_replay_max_prob=float(np.max(abs(p-reverse))),max_score_shift=float(np.max(abs(p-base))*100))
    candidate(tid,p,model,extra)

def load_acoustic():
    rows={}
    for path in (D/'acoustic_results').glob('band_*/evidence.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line);assert r['signal_id'] not in rows;rows[r['signal_id']]=r
    assert len(rows)==5189;return rows

def severity_features(records,field,wordmean=True):
    tr,va,_=parents();allx=[]
    for z in [tr,va]:
        x=[]
        for i,sid in enumerate(z['signal_ids']):
            raw=np.asarray(records[str(sid)][field],float);raw=raw.mean(0) if wordmean else raw;sev=int(z['severity'][i]);x.append(np.r_[raw,raw*(sev==1),raw*(sev==2)])
        allx.append(np.asarray(x))
    return allx

def acoustic():
    if not (D/'acoustic_complete.json').exists():return
    rows=load_acoustic();source=T.read(D/'acoustic_selection.json');none=[rows[r['signal_id']]['diagnostics']['high_vs_1_4_db'] for r in source if r['severity']==0];summary=dict(UTC=T.utc(),rows=5189,none_high_vs_1_4_db_quantiles=np.quantile(none,[.1,.5,.9]).tolist(),none_above_minus80_fraction=float(np.mean(np.asarray(none)>-80)),official_noise_version_not_identified=True,threshold_approximation_not_ISO=True)
    T.save(D/'Z1_physical_summary.json',summary)
    if summary['none_above_minus80_fraction']<.5:
        T.save(D/'Z1_measurement_stopped.json',dict(summary,reason='Most no-loss clips have negligible8-16kHz physical energy; original technical stop condition'))
        return
    fit_sentence('Z1',*severity_features(rows,'x'),[s+'_'+k for s in ['shared','mild','moderate'] for k in ['EHF_relative_level','EHF_mid_envelope_relation','approx_audible_fraction']],dict(summary=summary,protocol=T.read(D/'raw_acoustic_protocol.json')))

def word_proxy():
    path=D/'crossfit_word_proxy.npz'
    if path.exists():return G.P.F.arr(path)
    from v10_Y1_Y3_cached_20261022 import load
    tr,va,records=load();human=T.read(T.V/'final15h_20261022/Y1/rapid_beep_word_labels.json');align=G.alignment();features={};folds={}
    for sid,r in records.items():
        family=[]
        for fam in ['MF','MOSS']:
            src={s['stream']:s for s in r['sources'] if s['family']==fam}
            family.extend([np.asarray(align.align_words(' '.join(r['reference']),' '.join(src[s]['words']))['labels']) if s in src else np.zeros(len(r['reference'])) for s in ['A1L','A1R','A2L','A2R']])
        xx=np.stack(family,1);sev=r['severity'];features[sid]=np.column_stack([np.ones(len(xx)),xx,np.full(len(xx),sev==1),np.full(len(xx),sev==2)]);folds[sid]=int(hashlib.sha256(' '.join(r['reference']).encode()).hexdigest()[:8],16)%5
    models=[];weights={};training=list(map(str,tr['signal_ids']))
    for fold in range(5):
        use=[s for s in training if folds[s]!=fold and human[s]['hits_match']];x=np.concatenate([features[s] for s in use]);y=np.concatenate([human[s]['labels'] for s in use])
        def fun(w):
            u=x@w;return float(np.mean(np.logaddexp(0,u)-y*u)+.002*(w[1:]@w[1:])),x.T@(expit(u)-y)/len(y)+.004*np.r_[0,w[1:]]
        r=minimize(fun,np.zeros(11),jac=True,method='L-BFGS-B',options=dict(maxiter=250));assert r.success;models.append(r.x)
    full=np.asarray(T.read(T.V/'final15h_20261022/Y3/lexical_table.json')['word_base_coefficients']);width=max(len(x) for x in features.values());out={}
    for split,z in [('train',tr),('valid',va)]:
        proxy=np.full((len(z['signal_ids']),width),.5);mask=np.zeros_like(proxy,bool)
        for i,sid in enumerate(map(str,z['signal_ids'])):
            xx=features[sid];proxy[i,:len(xx)]=expit(xx@(models[folds[sid]] if split=='train' else full));mask[i,:len(xx)]=True
        out[split+'_proxy']=proxy;out[split+'_mask']=mask
    np.savez_compressed(path,**out);T.save(D/'word_proxy_provenance.json',dict(UTC=T.utc(),folds='5 SHA256 normalized-lyric groups, own group excluded',parameters_per_base=11,models=[m.tolist() for m in models],source_Y3_full_SHA=T.sha(T.V/'final15h_20261022/Y3/lexical_table.json'),not_exact_current_parent_OOF=True));return out

def anchor(proxy,mask,parent):
    l=logit(np.clip(proxy,1e-8,1-1e-8));lo=np.full(len(l),-50.);hi=np.full(len(l),50.);n=mask.sum(1)
    for _ in range(70):
        mid=(lo+hi)/2;p=expit(l+mid[:,None]);q=(p*mask).sum(1)/n;lo=np.where(q<parent,mid,lo);hi=np.where(q<parent,hi,mid)
    p=expit(l+((lo+hi)/2)[:,None]);assert np.max(abs((p*mask).sum(1)/n-parent))<1e-12;return p

def text():
    if not (D/'Z2_features_complete.json').exists() or (D/'Z2/candidate/verification.json').exists():return
    tr,va,_=parents();proxy=word_proxy();links=T.read(D/'model_row_links.json');records={}
    for file in (D/'model_results').glob('text_*/evidence.jsonl'):
        for line in file.read_text().splitlines():r=json.loads(line);assert r['text_key'] not in records;records[r['text_key']]=r
    arrays=[]
    for split,z in [('train',tr),('valid',va)]:
        mask=proxy[split+'_mask'];x=np.zeros((*mask.shape,2))
        for i,sid in enumerate(z['signal_ids']):
            row=np.asarray(records[links[str(sid)]['text_key']]['x']);assert len(row)==mask[i].sum();x[i,:len(row)]=row
        arrays.append(x)
    xm=proxy['train_mask'];mu=arrays[0][xm].mean(0);sd=np.maximum(arrays[0][xm].std(0),.01);a=np.clip((arrays[0]-mu)/sd,-5,5);b=np.clip((arrays[1]-mu)/sd,-5,5);trainp=anchor(proxy['train_proxy'],xm,tr['parent']);vm=proxy['valid_mask'];validp=anchor(proxy['valid_proxy'],vm,va['parent']);x=np.concatenate([np.ones((*xm.shape,1)),a],2);v=np.concatenate([np.ones((*vm.shape,1)),b],2);n=xm.sum(1);L=np.log1p(-np.clip(trainp,1e-12,1-1e-12));lam=.002
    def fun(w):
        t=np.tanh(x@w);k=np.exp(.2*t);eexp=np.exp(k*L);q=((1-eexp)*xm).sum(1)/n;err=q-tr['target'];dq=-eexp*L*k*.2*(1-t*t);grad=np.einsum('nl,nlk->nk',dq*xm,x)/n[:,None];return float(np.mean(err*err)+lam*(w@w)),np.mean(2*err[:,None]*grad,0)+2*lam*w
    opt=minimize(fun,np.zeros(3),jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-13,gtol=1e-9));assert opt.success,opt.message
    def predict(prob,mask,xx,w):return ((-np.expm1(np.exp(.2*np.tanh(xx@w))*np.log1p(-np.clip(prob,1e-12,1-1e-12))))*mask).sum(1)/mask.sum(1)
    p=predict(validp,vm,v,opt.x);zero=predict(validp,vm,v,np.zeros(3));reverse=predict(validp[::-1],vm[::-1],v[::-1],opt.x)[::-1];assert np.max(abs(zero-va['parent']))<1e-12 and np.max(abs(p-reverse))<1e-12
    candidate('Z2',p,dict(parameters=3,fields=['intercept','MLM_bidirectional_PLL','same_MLM_right_context_delta'],mean=mu.tolist(),std=sd.tolist(),weights=opt.x.tolist(),logk_bound=.2,regularization=lam),dict(mapping=T.read(D/'model_selection_complete.json'),word_proxy_provenance=T.read(D/'word_proxy_provenance.json'),zero_replay_max_prob=float(np.max(abs(zero-va['parent']))),reverse_replay_max_prob=float(np.max(abs(p-reverse))),new_head='bounded k-factor on sentence-anchored word probabilities',same_model_contrast=True))

def beat():
    if not (D/'Z3_features_complete.json').exists() or not (D/'acoustic_complete.json').exists():return
    records={}
    for file in (D/'model_results').glob('beat_*/evidence.jsonl'):
        for line in file.read_text().splitlines():r=json.loads(line);assert r['audio_key'] not in records;records[r['audio_key']]=r
    links=T.read(D/'model_row_links.json');rows=load_acoustic();sample=[records[k] for k in sorted(records)[:64]];usable=sum(r['diagnostics']['usable_words'] for r in sample);total=sum(r['diagnostics']['total_words'] for r in sample);coverage=usable/max(total,1);qualification=dict(UTC=T.utc(),clips=64,word_phase_coverage=coverage,minimum=.7,interval_CV_quantiles=np.quantile([r['diagnostics']['interval_CV'] for r in sample],[.1,.5,.9]).tolist(),manual_beat_truth_available=False,passed=coverage>=.7)
    T.save(D/'Z3_measurement_qualification.json',qualification)
    if not qualification['passed']:
        fit_sentence('Z4',*severity_features(rows,'Z4_edge',False),[s+'_'+edge for s in ['shared','mild','moderate'] for edge in ['first_energy','last_energy']],dict(beat_qualification=qualification,energy_only=True));return
    tr,va,_=parents();values=[]
    for z in [tr,va]:
        x=[]
        for i,sid in enumerate(z['signal_ids']):
            r=records[links[str(sid)]['audio_key']];q=np.mean(np.asarray(r['x']),0);edge=np.asarray(rows[str(sid)]['Z4_edge']);raw=np.r_[q,edge];sev=int(z['severity'][i]);x.append(np.r_[raw,raw*(sev==1),raw*(sev==2)])
        values.append(np.asarray(x))
    fit_sentence('Z3+Z4',*values,[s+'_'+f for s in ['shared','mild','moderate'] for f in ['beat_phase','downbeat','beat_confidence','first_energy','last_energy']],dict(beat_qualification=qualification,halfbeat_control_saved=True,edge_alignment='energy only'))

def poll():
    reg=T.read(D/'official_registry.json') if (D/'official_registry.json').exists() else dict(candidates=[])
    F.record=G.P.H.record
    for row in reg['candidates']:
        out=Path(row['candidate']);proof=out.parent/'official_complete.json'
        if proof.exists():continue
        spec=T.read(Path(row['manifest']));done=F.poll_official(Path(row['manifest']).parent,spec)
        if not done:continue
        r=T.read(out/'official_result.json');best=T.read(T.V/'current_best_own_valid.json');adopt=r['result']['RMSE']<best['RMSE']-1e-12
        if adopt:T.save(T.V/'current_best_own_valid.json',dict(best,UTC=T.utc(),submission_id=r['submission_id'],RMSE=r['result']['RMSE'],Correlation=r['result'].get('Correlation'),csv=r['csv'],csv_SHA=r['csv_SHA'],receipt=r['receipt'],previous_best=best,latest_task=row['task'],full_raw_infer_all_complete=False))
        T.save(proof,dict(UTC=T.utc(),official=r,adopted=adopt,previous_best_RMSE=best['RMSE']))
        tids=['Z3','Z4'] if row['task']=='Z3+Z4' else [row['task']]
        current={t['id'] for t in T.read(T.STATE)['tasks']}
        for tid in tids:
            if tid in current:T.mark_complete(tid,[str(proof),str(out/'verification.json')])
        print('Z_OFFICIAL',row['task'],r['submission_id'],r['result']['RMSE'],'ADOPT',adopt,flush=True)

def compose():
    marker=D/'positive_composition_decision.json'
    if marker.exists():return
    third=D/('Z3+Z4' if (D/'Z3+Z4/candidate/verification.json').exists() else 'Z4')/'candidate/official_result.json'
    if not third.exists() or not (D/'Z2/candidate/official_result.json').exists():return
    tr,va,parent=parents();sources=[]
    for tid in ['Z1','Z2','Z3+Z4','Z4']:
        path=D/tid/'candidate/official_result.json'
        if path.exists():
            result=T.read(path)
            if result['result']['RMSE']<parent['RMSE']-1e-9:sources.append((tid,result))
    if len(sources)<2:
        T.save(marker,dict(UTC=T.utc(),submitted=False,reason='Only one positive second-round correction; already adopted, do not duplicate its official submission',positive=[q[0] for q in sources]));return
    diffs=[]
    for tid,result in sources:
        ids,p=G.F.csvpred(result['csv']);assert np.array_equal(ids,va['signal_ids']);diffs.append((p-va['parent'])*100)
    delta=np.column_stack(diffs);gram=delta.T@delta/len(delta);q=np.array([r['result']['RMSE']**2-parent['RMSE']**2 for _,r in sources])-np.diag(gram)
    fit=minimize(lambda w:(float(q@w+w@gram@w),q+2*gram@w),np.ones(len(sources)),jac=True,bounds=[(.25,1)]*len(sources),method='L-BFGS-B');assert fit.success
    weights=np.clip(np.round(fit.x*4)/4,.25,1);forecast=float(np.sqrt(parent['RMSE']**2+q@weights+weights@gram@weights));best=T.read(T.V/'current_best_own_valid.json')
    decision=dict(UTC=T.utc(),sources=[dict(task=t,submission_id=r['submission_id'],RMSE=r['result']['RMSE']) for t,r in sources],weights=weights.tolist(),algebraic_RMSE_not_official=forecast,VALID_feedback_used=True,no_independent_generalization_claim=True,no_retraining=True)
    pred=va['parent']+delta@weights/100
    if forecast>=best['RMSE']-.0001 or np.min(pred)<0 or np.max(pred)>1:
        T.save(marker,dict(decision,submitted=False,reason='No useful bounded positive-only composition improvement, or out-of-range output; retain best confirmed system'));return
    candidate('Z-COMPOSE',pred,dict(parameters=len(weights),frozen_increment_weights=weights.tolist(),weights_selected_from='official aggregate MSE identities and prediction differences; rounded quarters',inherited_correction_parameters=sum(T.read(D/t/'model.json')['parameters'] for t,_ in sources)),decision)
    T.save(marker,dict(decision,submitted=True,candidate='Z-COMPOSE'))

def main():
    parents();T.save(D/'readout_pipeline_registered.json',dict(UTC=T.utc(),pid=os.getpid()));end=time.monotonic()+8*3600
    while time.monotonic()<end:
        for name,func in [('acoustic',acoustic),('text',text),('beat',beat),('official',poll),('composition',compose)]:
            try:func()
            except Exception as exc:T.save(D/(name+'_readout_attention.json'),dict(UTC=T.utc(),error=repr(exc),trace=traceback.format_exc()[-4000:]));print(name,traceback.format_exc(),flush=True)
        if (D/'positive_composition_decision.json').exists():
            decision=T.read(D/'positive_composition_decision.json');reg=T.read(D/'official_registry.json');all_done=all((Path(r['candidate']).parent/'official_complete.json').exists() for r in reg['candidates'])
            if all_done:T.save(D/'round_official_complete.json',dict(UTC=T.utc(),candidates=len(reg['candidates']),positive_composition=decision));return
        time.sleep(25)

if __name__=='__main__':main()
