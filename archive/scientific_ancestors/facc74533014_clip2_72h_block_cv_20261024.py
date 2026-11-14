"""One locked positive scale per new measurement; TRAIN-only fitting, no API."""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,csv,math
import numpy as np

V=Path(r'E:\icassp\zhaojiahao\CLIP2-new\algorithms\03_my_improvements\experiments\v10_20261016')
O=V/'next_gpu72h_20261024';D=O/'blocks';F=V/'next_fixed_blocks_20261024/FB05_artist/folds.npz'
P=V/'next_fixed_blocks_20261024/train_parent_audit/PRIOR_FIXED_train_conditional.npz'
ASSET=V/'next_fixed_blocks_20261024/history_asset_index.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def new(p,q):
    Path(p).parent.mkdir(exist_ok=True,parents=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(q,f,ensure_ascii=False,indent=2,allow_nan=False)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def vecsha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def axis(file,block,split,expected_ids):
    z=arrays(file);ids=z[split+'_signal_ids'].astype(str);lookup={s:i for i,s in enumerate(ids)}
    assert len(ids)==len(lookup) and all(s in lookup for s in expected_ids)
    ix=np.asarray([lookup[s] for s in expected_ids]);x=z[split+'_x'][ix].astype(float)
    if block=='A4':
        assert x.shape in [(len(ix),),(len(ix),1)];raw=x.reshape(-1)
    else:
        mask=z[split+'_mask'][ix].astype(bool);assert x.ndim==3 and x.shape[:2]==mask.shape and (mask.sum(1)>0).all()
        if block in ['A1','A1_ORIGINAL']:
            assert x.shape[2]==4;word=.5*((x[:,:,0]-x[:,:,1])+(x[:,:,2]-x[:,:,3]))
        elif block=='A2':assert x.shape[2]==2;word=x[:,:,0]-x[:,:,1]
        elif block=='A3':
            assert x.shape[2] in [1,2];word=x.mean(2)
        else:raise ValueError('Unknown fixed block')
        assert np.isfinite(word[mask]).all();raw=(np.where(mask,word,0).sum(1)/mask.sum(1))
    assert np.isfinite(raw).all();return raw
def standard(raw,use):
    mu=float(raw[use].mean());sd=max(float(raw[use].std()),1e-6)
    return np.clip((raw-mu)/sd,-5,5),mu,sd
def fit(s,r,weights=None):
    w=np.ones(len(s)) if weights is None else np.asarray(weights,float)
    den=float(np.sum(w*s*s));raw=float(np.sum(w*r*s)/den) if den>1e-12 else 0.
    return max(0.,raw),raw
def stats(coeff):
    mean=float(np.mean(coeff));se=float(np.std(coeff,ddof=1)/np.sqrt(5))
    zero=se<1e-12;z=(1e12 if mean>0 else 0.) if zero else mean/se
    return dict(mean=mean,SE=se,z=z,zero_SE=zero,shrink=max(0.,1-1/(z*z)) if z>=2 else 0.)
def register():
    D.mkdir(exist_ok=True)
    new(D/'protocol.json',dict(UTC=utc(),source=str(Path(__file__).resolve()),source_SHA=sha(__file__),
      folds=str(F),fold_SHA=sha(F),parent=str(P),parent_SHA=sha(P),scope='3072 exact conditional PF TRAIN rows; artist five folds frozen',
      raw_axes=dict(A1='mean_words(0.5*((MF_A1-MF_A2)+(MOSS_A1-MOSS_A2))), each variant best-ear SUM then six-variant equal mean',
       A1_ORIGINAL='Identical contrast with original-only cached G1a; TRAIN-only diagnostic comparator, never standalone official candidate',
       A2='mean_words(Qwen_bestear_SUM_A1-Qwen_bestear_SUM_A2)',
       A3='mean_words(Audio1_bestear_2x_SUM-1x_SUM); MF primary, fixed mean MF/MOSS iff MOSS technical controls pass; no CV family choice',
       A4='equal MF/MOSS and ears numeric expectation0to10, audio only',
       A6='20 same cluster-bootstrap one-scale learners:100*q_G1b-PF; nested fit-query provenance required'),
      preprocessing='Simple axes fit-fold mean/std (SD floor1e-6), clip standardized scalar to[-5,5]; no supervised intercept',
      scale='positive least squares through frozen PF residual; no grid search; five fold coefficients -> mean/SE; full TRAIN scale then shrink',
      gates=dict(z_min=2,heldout_gain_positive_folds_min=4,pooled_MSE_gain_positive=True,
        A1='Also no worse pooled artist CV MSE than original same-axis control',
        producer_technical_and_controls_also_required=True),
      SE_caveat='Five overlapping fit folds yield a heuristic z, not independent-fold inference or a formal guarantee',
      export='Frozen official2117 IDs order; native0to100 points; project to[0,100] and record out-of-range; no VALID labels',
      additional_own_virtual_views=0,no_official_action=True))
    print(json.dumps(dict(registered=True,source_SHA=sha(__file__))))
def evaluate(block,file,original=None):
    rule=read(D/'protocol.json');assert sha(__file__)==rule['source_SHA'] and sha(F)==rule['fold_SHA'] and sha(P)==rule['parent_SHA']
    dest=D/block;dest.mkdir(exist_ok=True);assert not (dest/'TRAIN_result.json').exists(),'Immutable CV already completed; reuse it'
    p=arrays(P);folds=arrays(F);ids=p['signal_ids'].astype(str);assert np.array_equal(ids,folds['signal_ids3072'].astype(str))
    base=p['prediction'].astype(float)*100;target=p['target'].astype(float)*100;fold=folds['fold_artist3072']
    raw=axis(file,block,'train',ids);prediction=np.full(len(ids),np.nan);coeff=[];normal=[]
    for k in range(5):
        use=fold!=k;held=fold==k;s,mu,sd=standard(raw,use);b,unconstrained=fit(s[use],target[use]-base[use]);coeff.append(b)
        prediction[held]=np.clip(base[held]+b*s[held],0,100);normal.append(dict(fold=k,mean=mu,std=sd,b=b,unconstrained_b=unconstrained))
    gain=(target-base)**2-(target-prediction)**2;gs=[float(gain[fold==k].mean()) for k in range(5)];st=stats(coeff)
    full_s,mu,sd=standard(raw,np.ones(len(ids),bool));b,braw=fit(full_s,target-base);shrunken=b*st['shrink']
    passed=bool(st['z']>=2 and sum(g>0 for g in gs)>=4 and gain.mean()>0)
    comparator=None
    if block=='A1':
        assert original,'A1 requires frozen original comparator'
        comparator=read(D/'A1_ORIGINAL/TRAIN_result.json');assert comparator['raw_vector_definition']=='A1_ORIGINAL'
        passed=passed and float(gain.mean())>=comparator['paired_MSE_gain']-1e-9
    np.savez_compressed(dest/'TRAIN_predictions.npz',signal_ids=ids,raw=raw,parent_score=base,target_score=target,
      OOF_candidate_score=prediction,outer_fold=fold,gain_MSE=gain,shrunken_full_delta_score=shrunken*full_s)
    result=dict(UTC=utc(),block=block,raw_vector_definition=block,fields=str(Path(file).resolve()),fields_SHA=sha(file),raw_vector_SHA=vecsha(raw),
      signal_ID_order_SHA=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),rows=3072,coefficients=normal,scale_statistics=st,
      paired_MSE_gain=float(gain.mean()),per_fold_MSE_gain=gs,positive_gain_folds=sum(g>0 for g in gs),
      full_mean=mu,full_std=sd,full_b=b,full_unconstrained_b=braw,shrunken_full_b=shrunken,
      raw_CV_gate_passed=passed,producer_gates_still_required=True,official_adoption_still_required=True,
      A1_original_comparator=comparator['paired_MSE_gain'] if comparator else None,
      source_SHA=rule['source_SHA'],fold_SHA=rule['fold_SHA'],parent_SHA=rule['parent_SHA'],
      strict_joint_upstream_OOF=False,new_VALID_virtual_views=0,no_official_action=True)
    new(dest/'TRAIN_result.json',result);print(json.dumps(result,ensure_ascii=False))
def export(block,file):
    rule=read(D/'protocol.json');assert sha(__file__)==rule['source_SHA']
    result=read(D/block/'TRAIN_result.json');assert result['raw_CV_gate_passed'] and block!='A1_ORIGINAL'
    a=read(ASSET)['assets']['PRIOR_FIXED'];assert sha(a['csv'])==a['SHA']
    rows=list(csv.reader(Path(a['csv']).open(encoding='utf-8-sig')));ids=np.asarray([r[0] for r in rows]);base=np.asarray([float(r[1]) for r in rows])
    raw=axis(file,block,'valid',ids);s=np.clip((raw-result['full_mean'])/result['full_std'],-5,5);unclip=base+result['shrunken_full_b']*s;prediction=np.clip(unclip,0,100)
    d=float(np.sqrt(np.mean((prediction-base)**2)));threshold=max(.005,1.5*d/np.sqrt(2117));dest=D/block/'candidate';dest.mkdir()
    path=dest/'valid_predictions.csv'
    with path.open('x',encoding='utf-8',newline='') as f:csv.writer(f,lineterminator='\n').writerows(zip(ids,[format(x,'.10f') for x in prediction]))
    report=dict(UTC=utc(),block=block,reference_submission=587334,reference_RMSE=a['RMSE'],base_SHA=a['SHA'],
      CSV_SHA=sha(path),d_points=d,ladder_threshold_RMSE=threshold,out_of_range_before_projection=int(np.sum((unclip<0)|(unclip>100))),
      expected_TRAIN_paired_MSE_gain=result['paired_MSE_gain'],expected_VALID_RMSE_gain_heuristic=result['paired_MSE_gain']/(2*a['RMSE']),
      expectation_not_official_score=True,CSV_values0to100=True,one_positive_scale=True,no_VALID_labels=True,
      producer_qualification_must_be_bound_before_official=True,new_virtual_views=0,TRAIN_result_SHA=sha(D/block/'TRAIN_result.json'),
      no_production_adoption=True,no_official_action=True)
    new(dest/'report.json',report);new(dest/'verification.json',dict(status='verified',csv_sha256=sha(path),report_sha256=sha(dest/'report.json')))
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','evaluate','export']);ap.add_argument('--block',choices=['A1','A1_ORIGINAL','A2','A3','A4']);ap.add_argument('--fields');ap.add_argument('--original');a=ap.parse_args()
    if a.action=='register':register()
    elif a.action=='evaluate':assert a.block and a.fields;evaluate(a.block,a.fields,a.original)
    else:assert a.block and a.fields;export(a.block,a.fields)
