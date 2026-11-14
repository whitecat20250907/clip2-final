"""Third round: fixed evidence, prompt-group cross-fitting, six-slot window."""
from pathlib import Path
import os,sys,json,csv,hashlib,collections,datetime as dt
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'v10_final15h_site')]
import v10_new_information_20261022 as G
import v10_Z_readout_20261023 as Z
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
T=G.T;F=G.P.H.F;D=T.V/'third_round_20261023';D.mkdir(exist_ok=True)
DEADLINE='2026-10-23T11:01:20+00:00'

def initialize():
    if not (D/'execution_protocol.json').exists():
        source=Path(r'C:\Users\zjh13\.codex\attachments\2701314d-d62d-418f-baef-d437962d3439\Pasted text.txt')
        (D/'source_plan.txt').write_bytes(source.read_bytes())
        T.save(D/'own_reference.json',T.read(T.V/'current_best_own_valid.json'))
        T.save(D/'joint_reference.json',T.read(T.V/'current_best_joint_valid.json'))
        T.save(D/'execution_protocol.json',dict(UTC=T.utc(),start_UTC='2026-10-23T07:01:20+00:00',deadline_UTC=DEADLINE,source_plan_SHA=T.sha(source),own_limit=5,total_limit=6,reserve_last_for_joint=True,own_selection_cutoff_UTC='2026-10-23T10:10:00+00:00',joint_submit_UTC='2026-10-23T10:25:00+00:00',new_params_max=8,VALID_labels=False,EVAL=False,TRAIN_parent='Inherited ER01 conditional surrogate; 5-fold word proxy verified; full recursively updated winner OOF not established',routes=['Z6','Z11','Z7','Z8','Z12','Z13','Z9'],conditional=['Z10'],selection='TRAIN prompt-group crossfit ranks technically valid fixed candidates, controls never submitted; reserve own fifth for positive increment composition if useful'))
        T.save(D/'official_registry.json',dict(candidates=[]))
        titles={'Z6':'词汇知识常模＋现有Y3词项证据','Z11':'词尾形态与听损交互','Z7':'序列位置与记忆负荷','Z8':'掩码语境竞争词','Z12':'异质听者的共识误听','Z13':'CLIP1真实人类词项先验','Z9':'PANNs背景人声场景证据'}
        with T.locked() as ok:
            assert ok
            st=T.read(T.STATE);known={r['id'] for r in st['tasks']}
            for k,title in titles.items():
                if k not in known:st['tasks'].append(dict(id=k,task=title,state='已启动：第三轮4小时窗口',ETA='首轮候选0.5–3小时；官测截至04:01 PDT',dependency='技术核验、外部资源或GPU就绪',source_id='第三轮计划20261023'))
            T.save(T.STATE,st)
        T.refresh()
    return T.read(D/'execution_protocol.json')

def load():
    from v10_Y1_Y3_cached_20261022 import load as load0
    tr,va,records=load0();ref=T.read(D/'own_reference.json');ids,p=G.F.csvpred(ref['csv']);assert np.array_equal(ids,va['signal_ids']);va['parent']=p
    proxy=Z.word_proxy();tm=proxy['train_mask'];vm=proxy['valid_mask']
    tp=Z.anchor(proxy['train_proxy'],tm,tr['parent']);vp=Z.anchor(proxy['valid_proxy'],vm,va['parent'])
    folds=np.array([int(hashlib.sha256(' '.join(records[str(s)]['reference']).encode()).hexdigest()[:8],16)%5 for s in tr['signal_ids']])
    return tr,va,records,tm,vm,tp,vp,folds

def fit_route(tid,train,valid,fields,extra,control=None,transform='logit'):
    out=D/tid;out.mkdir(exist_ok=True)
    if (out/'ready.json').exists():return
    tr,va,recs,tm,vm,tp,vp,fold=load();a=np.asarray(train,float);b=np.asarray(valid,float)
    assert a.shape==(*tm.shape,len(fields)) and b.shape==(*vm.shape,len(fields)) and len(fields)<=8
    assert np.isfinite(a).all() and np.isfinite(b).all()
    mu=a[tm].mean(0);sd=np.maximum(a[tm].std(0),.01);a=np.clip((a-mu)/sd,-5,5);b=np.clip((b-mu)/sd,-5,5)
    # No free intercept: zero coefficients exactly recover the frozen parent.
    L=logit(np.clip(tp,1e-9,1-1e-9));LV=logit(np.clip(vp,1e-9,1-1e-9));n=tm.sum(1);lam=.0002
    def predict(base,x,m,w):
        v=.2*np.tanh(x@w)
        q=-np.expm1(np.exp(v)*np.log1p(-np.clip(expit(base),1e-12,1-1e-12))) if transform=='power' else expit(base+v)
        return (q*m).sum(1)/m.sum(1)
    def fit(use,xx=a):
        x=xx[use];l=L[use];m=tm[use];nn=n[use];y=tr['target'][use]
        def fun(w):
            v=np.tanh(x@w)
            if transform=='power':
                logfail=np.log1p(-np.clip(expit(l),1e-12,1-1e-12));k=np.exp(.2*v);f=np.exp(k*logfail);q=1-f;dq=-f*logfail*k*.2*(1-v*v)*m
            else:q=expit(l+.2*v);dq=.2*q*(1-q)*(1-v*v)*m
            e=(q*m).sum(1)/nn-y;grad=np.einsum('nl,nlk->nk',dq,x)/nn[:,None]
            return float(np.mean(e*e)+lam*(w@w)),np.mean(2*e[:,None]*grad,0)+2*lam*w
        r=minimize(fun,np.zeros(len(fields)),jac=True,method='L-BFGS-B',options=dict(maxiter=350,ftol=1e-13,gtol=1e-9));assert r.success,r.message;return r.x
    oof=np.zeros(len(tp));ws=[]
    for k in range(5):
        w=fit(fold!=k);ws.append(w.tolist());use=fold==k;oof[use]=predict(L[use],a[use],tm[use],w)
    base=tr['parent'];delta=oof-base;y=tr['target'];scores={str(float(s)):float(np.mean((base+s*delta-y)**2)*10000) for s in [0,.5,1]}
    scale=float(min(scores,key=scores.get));w=fit(np.ones(len(tp),bool));raw=predict(LV,b,vm,w);p=va['parent']+scale*(raw-va['parent']);zero=predict(LV,b,vm,np.zeros(len(fields)))
    assert np.max(abs(zero-va['parent']))<1e-8
    assert np.max(abs(p-(va['parent'][::-1]+scale*(predict(LV[::-1],b[::-1],vm[::-1],w)-va['parent'][::-1]))[::-1]))<1e-12
    if control is not None:
        cc=np.asarray(control,float);assert cc.shape==train.shape;cc=np.clip((cc-mu)/sd,-5,5);coof=np.zeros(len(tp))
        for k in range(5):
            cw=fit(fold!=k,cc);use=fold==k;coof[use]=predict(L[use],cc[use],tm[use],cw)
        extra['control_OOF_MSE']=float(np.mean((coof-y)**2)*10000)
    model=dict(parameters=len(fields),fields=fields,mean=mu.tolist(),std=sd.tolist(),weights=w.tolist(),fold_weights=ws,scale=scale,logit_bound=.2,transform=transform,regularization=lam,selection='0/half/full fixed by 5-fold normalized lyric-group TRAIN crossfit',TRAIN_scope='3072 inherited conditional sentence parents, not exact current winner nested OOF')
    np.savez_compressed(out/'features.npz',train=train,valid=valid,train_signal_ids=tr['signal_ids'],valid_signal_ids=va['signal_ids']);np.savez_compressed(out/'oof.npz',signal_ids=tr['signal_ids'],parent=base,target=y,pred=oof,folds=fold)
    T.save(out/'model.json',model);c=out/'candidate';c.mkdir(exist_ok=True)
    with (c/'valid_predictions.csv').open('w',newline='') as f:csv.writer(f).writerows(zip(va['signal_ids'],[format(float(q)*100,'.10f') for q in p]))
    report=dict(UTC=T.utc(),task=tid,csv_SHA=T.sha(c/'valid_predictions.csv'),model_SHA=T.sha(out/'model.json'),frozen_parent=T.read(D/'own_reference.json'),TRAIN=3072,VALID=2117,VALID_labels=False,EVAL=False,zero_replay=float(np.max(abs(zero-va['parent']))),OOF_MSE_by_scale=scores,OOF_gain_MSE=scores['0.0']-scores[str(scale)],OOF_fold_gain_MSE=[float(np.mean((base[fold==k]-y[fold==k])**2-(base[fold==k]+scale*delta[fold==k]-y[fold==k])**2)*10000) for k in range(5)],delta_RMS_score=float(np.sqrt(np.mean((p-va['parent'])**2))*100),extra=extra)
    T.save(c/'report.json',report);T.save(c/'verification.json',dict(status='verified',csv_sha256=report['csv_SHA'],report_sha256=T.sha(c/'report.json')))
    T.save(out/'ready.json',dict(UTC=T.utc(),task=tid,scale=scale,gain=report['OOF_gain_MSE'],candidate=str(c),fields=len(fields)));print('ROUND3_READY',tid,'scale',scale,'OOFgain',report['OOF_gain_MSE'],flush=True)

def progress(tid,state,eta):
    with T.locked() as ok:
        if not ok:return
        st=T.read(T.STATE)
        for r in st['tasks']:
            if r['id']==tid:r.update(state=state,ETA=eta)
        T.save(T.STATE,st)
    T.refresh()
