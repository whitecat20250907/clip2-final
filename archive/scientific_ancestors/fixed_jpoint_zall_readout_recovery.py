"""Two fixed late-refreeze readouts; explicit new authorization is mandatory.

Audit checks bytes, source lineage and complete inputs only. Execution fits
each predeclared fullTRAIN positive scale once, reports original artist CV,
and writes VALID CSVs. No API, GPU, scheduler or EVAL submission output.
"""
import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

CONTRACT_SHA = '5a6bdea070fd3b9c52f34472b24e1d0a2480b93a972ea60701dccbb4fcdbf5e5'
DEFINITION_SHA = '6a49cdfdf1120ae54ae3cdd4ecd65a96de99984d9e89ee1933e5e0b94d64d5f1'
A4_CODE_SHA = 'facc74533014125ac3d59862fab21d9557b7ca1676de2439d84a20b5cf11522f'
PF_CSV_SHA = 'a32b857ca9bf1da0bb4dcdcdf215bd4034ed74a7bfa060865b100d35b2209b74'
PARENT_SHA = 'afc69a085b9c0ee4ead6fdce0de5f72738771b03f35b88fccf2e58c1a7ab081f'
FOLD_SHA = '19c75635f4fd3c830321cbd50a287c53e9b52ddaeb99c380461f50a14888988f'
CONTROL_SHA = '0360fb00e0f89aa1f0706e0b381286d148d94f1cc0f791bcc8c5b442b0247b5d'
LATE_AUTH_SHA = '4eaf1a35bd9f6f503ed1c828cc5eefa688021ef43c60b0a3d9e16b0c65aa1975'
JUDGES = ['Qwen25Omni7B', 'MiDashengLM7B']
CANDIDATES = ['J-point', 'Z-all']
EXPECTED_PROMPTS = ['Q1_original', 'Q1_rewrite1', 'Q1_rewrite2', 'Q2_vocal_foreground', 'Q2_articulation', 'Q2_masking_negative']


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def canonical(q): return json.dumps(q, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()
def vector_sha(x): return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def checked(ref):
    p=Path(ref['path']).resolve(); assert sha(p)==ref['SHA'],str(p)
    return p
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def immutable(path,q):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    try:
        with tmp.open('xb') as f:f.write(canonical(q)+b'\n');f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
def save_npz(path,**q):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz')
    try:
        np.savez_compressed(tmp,**q)
        with tmp.open('rb+') as f:os.fsync(f.fileno())
        os.link(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


@contextlib.contextmanager
def exclusive_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b');f.seek(0,2)
    if f.tell()==0:f.write(b'0');f.flush()
    f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        f.close()


def standard(raw,use):
    """Identical numerical scalar transform to frozen C.standard."""
    mu=float(raw[use].mean());sd=max(float(raw[use].std()),1e-6)
    return np.clip((raw-mu)/sd,-5,5),mu,sd


def positive_fit(s,r):
    """Identical unweighted through-origin formula to frozen C.fit."""
    den=float(np.sum(s*s));unconstrained=float(np.sum(r*s)/den) if den>1e-12 else 0.
    return max(0.,unconstrained),unconstrained


def z_train(x,n=3072):
    mu=np.asarray(x[:n].mean(axis=0),np.float64)
    sd=np.maximum(np.asarray(x[:n].std(axis=0,ddof=0),np.float64),1e-6)
    return (x-mu)/sd,mu,sd


def auth_check(c,check_time=True):
    auth=read(checked(c['authorization']))
    assert c['authorization']['SHA']==LATE_AUTH_SHA
    assert auth['execution_authorized'] is True and auth['user_answer_received'] is True
    assert auth['stage_ID']==c['stage_ID']
    assert auth['candidate_freeze_epoch']==c['cutoff_epoch']
    assert auth['allowed_candidates']==CANDIDATES
    for key in ['permit_fullTRAIN_positive_OLS_once','permit_artist_CV_diagnostic_once','permit_VALID_CSV_creation']:
        assert auth[key] is True,key
    assert c['cutoff_epoch']<=1793032803.0, 'No extension beyond explicit09:40:03PDT'
    if check_time:assert time.time()<c['cutoff_epoch'], 'Late-refreeze cutoff reached'
    return auth


def prepare(args):
    assert sha(args.config)==args.config_sha
    c=read(args.config);assert c['schema']=='CLIP2.jury.late_fixed_readout.config.v1'
    assert c['producer_SHA']==sha(__file__)
    assert c['candidates']==CANDIDATES and c['judges']==JUDGES and c['B_angles']==['B3_AUDIOBOX']
    source=checked(c['source_contract']);assert sha(source)==CONTRACT_SHA
    definition=checked(c['definition_authorization']);assert sha(definition)==DEFINITION_SHA
    original_code=checked(c['source_A4_code']);assert sha(original_code)==A4_CODE_SHA
    control=read(checked(c['control_binding']));assert c['control_binding']['SHA']==CONTROL_SHA
    assert control['allowed_candidates']==CANDIDATES and control['roster_point']==JUDGES and control['roster_B']==['B3_AUDIOBOX']
    assert control['no_new_gate_prompt_field_weight_search'] is True
    point=read(checked(c['point_protocol']));inp=read(checked(c['input_manifest']))
    binding=read(checked(c['input_binding']))
    assert binding['schema']=='CLIP2.jury.late_readout.input_binding.v1'
    assert point['no_word_axis'] is True and len(point['rows'])==7378 and len(inp['rows'])==7378
    assert point['prompts_SHA']==hashlib.sha256(canonical(point['prompts'])).hexdigest()
    assert [x['id']for x in point['prompts']]==EXPECTED_PROMPTS
    assert point['prompts'][-1]['direction']=='10-minus-expectation'
    ids=np.asarray([r['signal_id']for r in point['rows']]);splits=np.asarray([r['split']for r in point['rows']])
    assert binding['signal_ids']==ids.tolist() and binding['split']==splits.tolist()
    assert binding['rows']==point['rows'] and binding['rows_SHA']==point['rows_SHA']
    assert binding['checks']['jury_has_no_word_axis'] is True
    assert len(set(ids))==7378 and np.array_equal(splits,np.asarray(['train']*3072+['valid']*2117+['eval']*2189))
    assert [(r['signal_id'],r['split'],r['audio_path'])for r in point['rows']]==[(r['signal_id'],r['split'],r['audio_path'])for r in inp['rows']]
    pfile=checked(c['parent']);ffile=checked(c['folds']);vfile=checked(c['PF_VALID'])
    assert sha(pfile)==PARENT_SHA and sha(ffile)==FOLD_SHA and sha(vfile)==PF_CSV_SHA
    parent=arrays(pfile);folds=arrays(ffile)
    assert np.array_equal(ids[:3072],parent['signal_ids'].astype(str)) and np.array_equal(ids[:3072],folds['signal_ids3072'].astype(str))
    assert hashlib.sha256(canonical(parent['lyric_sha256'].astype(str).tolist())).hexdigest()==point['TRAIN_lyric_SHA_sequence_SHA']
    assert hashlib.sha256(canonical(parent['audio2_sha256'].astype(str).tolist())).hexdigest()==point['TRAIN_Audio2_SHA_sequence_SHA']
    assert binding['TRAIN3072_lyric_SHA_list']==parent['lyric_sha256'].astype(str).tolist()
    assert binding['TRAIN3072_Audio2_SHA_list']==parent['audio2_sha256'].astype(str).tolist()
    fold=folds['fold_artist3072'];assert fold.shape==(3072,) and set(fold.tolist())=={0,1,2,3,4}
    csvrows=list(csv.reader(vfile.open(encoding='utf-8-sig',newline='')))
    assert len(csvrows)==2117 and all(len(r)==2 for r in csvrows)
    assert [r[0]for r in csvrows]==ids[3072:5189].tolist()
    basev=np.asarray([float(r[1])for r in csvrows],np.float64)
    assert np.isfinite(basev).all() and ((0<=basev)&(basev<=100)).all()
    axis={};lineage={}
    for judge in JUDGES:
        spec=c['axes'][judge];rawfile=checked(spec['raw']);complete=read(checked(spec['complete']));gfile=checked(spec['TRAIN_gate']);gate=read(gfile)
        assert complete['complete'] is True and complete['judge']==judge and complete['rows']==7378 and complete['measurements']==88536
        assert complete['raw_fields_eligible'] is True and complete['raw_fields_SHA']==sha(rawfile)
        assert complete['TRAIN_gate_SHA']==sha(gfile) and gate['TRAIN_gate_passed'] is True
        assert gate['judge']==judge and complete['identity_SHA']==gate['identity_SHA']
        assert complete['stage_config_SHA']==gate['stage_config_SHA']==spec['execution_config_SHA']
        assert complete['execution_stage_ID']==gate['execution_stage_ID']==spec['execution_stage_ID']
        assert complete['old_epoch_candidate_eligible'] is False and complete['produced_before_original_T18'] is False
        assert complete['TRAIN_gate_reused_without_recomputation'] is True
        assert gate['SD_point3_gate'] is True and gate['severity_gate'] is True and gate['TRAIN_raw6mean_SD_population']>.3
        assert gate['TRAIN_severity_raw6mean'][0]>gate['TRAIN_severity_raw6mean'][1]>gate['TRAIN_severity_raw6mean'][2]
        assert not any(Path(r).exists()for r in spec.get('scientific_failure_paths',[])), 'Roster seat closed'
        assert not any(Path(r).exists()for r in spec.get('engineering_attention_paths',[])), 'Engineering attention'
        z=arrays(rawfile)
        assert np.array_equal(z['signal_ids'].astype(str),ids) and np.array_equal(z['split'].astype(str),splits)
        assert z['prompt_ids'].astype(str).tolist()==EXPECTED_PROMPTS
        assert z['score_ears'].shape==(7378,6,2) and z['raw_prompt_fields'].shape==(7378,6) and z['raw6mean'].shape==(7378,)
        assert np.isfinite(z['score_ears']).all() and np.all((z['score_ears']>=0)&(z['score_ears']<=10))
        assert np.array_equal(z['raw_prompt_fields'],z['score_ears'].mean(2)) and np.array_equal(z['raw6mean'],z['raw_prompt_fields'].mean(1))
        axis[judge]=z['raw_prompt_fields'].astype(np.float64)
        lineage[judge]=dict(raw=spec['raw'],complete=spec['complete'],TRAIN_gate=spec['TRAIN_gate'],identity_SHA=complete['identity_SHA'],stage_config_SHA=complete['stage_config_SHA'],numeric_orientation='All raw fields already oriented; negative masking was10-E at measurement. Never invert again.')
    bs=c['B3'];brfile=checked(bs['raw']);bc=read(checked(bs['complete']));b=arrays(brfile)
    assert bc['raw_axes_eligible'] is True and bc['TRAIN_PQ_severity_gate'] is True and bc['rows']==7378 and bc['measurements']==14756
    assert bc['raw_axes_SHA']==sha(brfile)
    assert np.array_equal(b['signal_ids'].astype(str),ids) and np.array_equal(b['split'].astype(str),splits)
    assert b['B3_raw'].shape==(7378,) and np.isfinite(b['B3_raw']).all()
    assert b['PQ_ears'].shape==b['PC_ears'].shape==(7378,2)
    assert np.array_equal(b['PQ_mean'],b['PQ_ears'].mean(1)) and np.array_equal(b['PC_mean'],b['PC_ears'].mean(1))
    assert np.array_equal(b['TRAIN_mean_PQ_PC'],np.asarray(bc['TRAIN_mean_PQ_PC'])) and np.array_equal(b['TRAIN_SD_population_PQ_PC'],np.asarray(bc['TRAIN_SD_population_PQ_PC']))
    fixed=(b['PQ_mean']-b['TRAIN_mean_PQ_PC'][0])/b['TRAIN_SD_population_PQ_PC'][0]-(b['PC_mean']-b['TRAIN_mean_PQ_PC'][1])/b['TRAIN_SD_population_PQ_PC'][1]
    assert np.array_equal(b['B3_raw'],fixed), 'Use only original frozen B3 axis'
    output=Path(c['output_dir']).resolve()
    assert c['stage_ID']=='JURY_LATE_TWO_20261026' and output.name=='late_two_candidates_serialization_20261026'
    if c.get('original_A4_axis'):
        oldref=c['original_A4_axis'];az=arrays(checked(oldref));ai=az['train_signal_ids'].astype(str)
        assert len(ai)==len(set(ai))==3072 and set(ai)==set(ids[:3072])
        lookup={s:i for i,s in enumerate(ai)};ar=az['train_x'][[lookup[s]for s in ids[:3072]]].reshape(-1)
        assert ar.shape==(3072,) and np.isfinite(ar).all()
        assert vector_sha(ar)==binding['old_A4']['TRAIN_raw_vector_SHA_after_ID_reorder']
    return c,ids,splits,parent,fold,basev,axis,b['B3_raw'].astype(np.float64),lineage


def build_fixed_raw(axis,b3):
    judgeaxes={};norm={}
    for judge in JUDGES:
        standardized,mu,sd=z_train(axis[judge]);judgeaxes[judge]=standardized.mean(1)
        norm[judge]=dict(question_TRAIN_means=mu.tolist(),question_TRAIN_population_SD=sd.tolist(),question_weights=[1/6]*6)
    jpoint=(judgeaxes[JUDGES[0]]+judgeaxes[JUDGES[1]])/2
    bz,bmu,bsd=z_train(b3)
    norm['B3']=dict(derived_axis_TRAIN_mean=float(bmu),derived_axis_TRAIN_population_SD=float(bsd))
    return {'J-point':jpoint,'Z-all':(jpoint+bz)/2},norm,judgeaxes


def execute(args,prepared):
    c,ids,splits,parent,fold,basev,axis,b3,lineage=prepared
    auth=auth_check(c,check_time=False);out=Path(c['output_dir']).resolve()
    with exclusive_lock(out/'fit.lock'):
        completefile=out/'complete.json'
        if completefile.exists():
            q=read(completefile);assert q['stage_config_SHA']==args.config_sha and q['producer_SHA']==sha(__file__)
            for f in q['artifacts']:assert sha(f['path'])==f['SHA']
            print(json.dumps(dict(reused_complete=True,path=str(completefile),SHA=sha(completefile))));return
        assert not(out/'fit_intent.json').exists(), 'Existing unknown/partial fit intent; no retry'
        auth_check(c)
        immutable(out/'fit_intent.json',dict(UTC=utc(),stage_ID=c['stage_ID'],stage_config_SHA=args.config_sha,producer_SHA=sha(__file__),authorization=c['authorization'],fixed_candidates=CANDIDATES,fullTRAIN_fit_count_per_candidate=1,diagnostic_CV_count_per_candidate=1,no_retry_on_unknown=True,no_API_or_EVAL_submission=True))
        try:
            raw,norm,judgeaxes=build_fixed_raw(axis,b3)
            base=parent['prediction'].astype(np.float64)*100;target=parent['target'].astype(np.float64)*100
            assert np.isfinite(base).all() and np.isfinite(target).all()
            artifacts=[];reports={};written_CSV_hashes={};trainuse=np.arange(7378)<3072
            for name in CANDIDATES:
                assert time.time()<c['cutoff_epoch'], 'Refreeze deadline reached; preserve partial'
                r=raw[name];assert r.shape==(7378,) and np.isfinite(r).all()
                final,mu,sd=standard(r,trainuse)
                coefficient,unconstrained=positive_fit(final[:3072],target-base)
                # FullTRAIN raw question/B3 transforms stay frozen for all
                # five report-only folds. Only final scalar transform and
                # positive scale are estimated on each fit fold.
                cv=np.full(3072,np.nan);coefficients=[]
                for k in range(5):
                    fit=np.flatnonzero(fold!=k);held=np.flatnonzero(fold==k)
                    transformed,fold_mu,fold_sd=standard(r[:3072],fold!=k)
                    bk,braw=positive_fit(transformed[fit],(target-base)[fit])
                    cv[held]=np.clip(base[held]+bk*transformed[held],0,100)
                    coefficients.append(dict(fold=k,fit_rows=int(len(fit)),held_rows=int(len(held)),scalar_mean=fold_mu,scalar_SD=fold_sd,positive_b=bk,unconstrained_b=braw))
                assert np.isfinite(cv).all()
                gain=(target-base)**2-(target-cv)**2
                dest=out/name;dest.mkdir(parents=True,exist_ok=True)
                trainfile=dest/'TRAIN_diagnostic.npz'
                save_npz(trainfile,signal_ids=ids[:3072],raw_fixed_axis=r[:3072],parent_score=base,target_score=target,artist_fold=fold,OOF_candidate_score=cv,paired_MSE_gain=gain,fullTRAIN_standardized_axis=final[:3072])
                unprojected=basev+coefficient*final[3072:5189];prediction=np.clip(unprojected,0,100)
                delta=prediction-basev;d=float(np.sqrt(np.mean(delta*delta)));ladder=max(.005,1.5*d/math.sqrt(2117))
                deltafile=dest/'VALID_delta.npz'
                save_npz(deltafile,signal_ids=ids[3072:5189],PRIOR_FIXED_score=basev,candidate_score=prediction,delta_score=delta)
                distinct=coefficient>0 and bool(np.any(delta!=0))
                csvfile=None
                if distinct:
                    csvfile=dest/'valid_predictions.csv'
                    with csvfile.open('x',encoding='utf-8',newline='')as f:csv.writer(f,lineterminator='\n').writerows(zip(ids[3072:5189],[format(float(x),'.10f')for x in prediction]))
                    written=list(csv.reader(csvfile.open(encoding='utf-8',newline='')))
                    assert [x[0]for x in written]==ids[3072:5189].tolist() and len(written)==2117
                    assert np.max(np.abs(np.asarray([float(x[1])for x in written])-prediction))<=5.01e-11
                csv_hash=sha(csvfile)if csvfile else None
                duplicate_of=written_CSV_hashes.get(csv_hash)if csv_hash else None
                if csv_hash and duplicate_of is None:written_CSV_hashes[csv_hash]=name
                report=dict(UTC=utc(),candidate=name,stage_ID=c['stage_ID'],producer_SHA=sha(__file__),stage_config_SHA=args.config_sha,authorization=c['authorization'],source_contract=c['source_contract'],
                    finalized_roster=JUDGES,B_angles=['B3_AUDIOBOX'] if name=='Z-all' else [],question_weights=[1/6]*6,judge_weights=[.5,.5],composition_weights=[.5,.5] if name=='Z-all' else None,
                    fullTRAIN_rows=3072,fullTRAIN_positive_OLS_b=coefficient,unconstrained_b=unconstrained,fullTRAIN_scalar_mean=mu,fullTRAIN_scalar_SD=sd,fullTRAIN_fit_count=1,no_intercept_or_shrink=True,
                    diagnostic_CV=dict(fold_protocol='Original fixed artist5fold',folds=c['folds'],coefficients=coefficients,paired_MSE_gain=float(gain.mean()),per_fold_MSE_gain=[float(gain[fold==k].mean())for k in range(5)],CV_selection_gate_applied=False,CV_shrink_applied=False,feature_normalization='FullTRAIN unsupervised question/B3 normalizations freeze raw axes before report-only scalar CV',conditional_PF_parent=True,strict_joint_nested_OOF=False,not_an_unbiased_strictly_nested_generalization_estimate=True),
                    raw_vector_SHA=vector_sha(r),TRAIN_raw_vector_SHA=vector_sha(r[:3072]),VALID_raw_vector_SHA=vector_sha(r[3072:5189]),
                    distinct_candidate=distinct,zero_coefficient_no_submission=coefficient==0,duplicate_of=duplicate_of,official_submission_eligible=bool(distinct and duplicate_of is None),CSV_SHA=csv_hash,CSV_path=str(csvfile)if csvfile else None,CSV_rows=2117 if csvfile else 0,
                    challenge=2703,phase=5346,private=True,reference_submission=587334,reference_RMSE=15.50577990350711,reference_CSV_SHA=PF_CSV_SHA,
                    d_points=d,ladder_threshold_RMSE=ladder,out_of_range_before_projection=int(np.sum((unprojected<0)|(unprojected>100))),CSV_values0to100=True,
                    VALID_labels_read=False,EVAL_prediction_or_submission_CSV_created=False,virtual_views=0,official_POSTs=0,no_production_adoption=True)
                rp=dest/'report.json';immutable(rp,report);reports[name]=dict(path=str(rp),SHA=sha(rp),distinct_candidate=distinct)
                for f in [trainfile,deltafile,rp]+([csvfile]if csvfile else[]):artifacts.append(dict(path=str(f),SHA=sha(f)))
            assert time.time()<c['cutoff_epoch'], 'Refreeze deadline passed; partial outputs not qualified frozen candidates'
            binding=dict(UTC=utc(),stage_ID=c['stage_ID'],stage_config_SHA=args.config_sha,authorization=c['authorization'],producer_SHA=sha(__file__),
                definitions=c['source_contract'],control_binding=c['control_binding'],source_A4_code=c['source_A4_code'],axes=lineage,B3=c['B3'],parent=c['parent'],folds=c['folds'],PF_VALID=c['PF_VALID'],point_protocol=c['point_protocol'],input_manifest=c['input_manifest'],input_binding=c['input_binding'],
                rows=7378,splits=dict(TRAIN=3072,VALID=2117,EVAL=2189),IDs_SHA=hashlib.sha256(canonical(ids.tolist())).hexdigest(),
                words_and_masks=dict(no_word_axis=True,meaning='Whole Audio1 scalar measurements. No word/token mask created or fitted.',TRAIN_lyric_SHA_sequence_SHA=read(checked(c['point_protocol']))['TRAIN_lyric_SHA_sequence_SHA'],TRAIN_audio2_SHA_sequence_SHA=read(checked(c['point_protocol']))['TRAIN_Audio2_SHA_sequence_SHA']),
                preprocessing=norm,judge_TRAIN_correlation=float(np.corrcoef(judgeaxes[JUDGES[0]][:3072],judgeaxes[JUDGES[1]][:3072])[0,1]),
                original_A4_correlation_status='Not calculated unless separately source-bound original axis supplied; absence is recorded, never fabricated',
                late_refreeze_after_original_T18=True,original_epoch_unchanged=True,no_official_API_EVAL_submission_virtual_weight_search=True)
            if c.get('original_A4_axis'):
                af=checked(c['original_A4_axis']);az=arrays(af)
                ai=az['train_signal_ids'].astype(str);lookup={s:i for i,s in enumerate(ai)}
                ar=az['train_x'][[lookup[s]for s in ids[:3072]]].reshape(-1)
                binding['original_A4_axis']=c['original_A4_axis'];binding['original_A4_raw_vector_SHA_after_ID_reorder']=vector_sha(ar)
                binding['Jpoint_TRAIN_correlation_with_original_A4']=float(np.corrcoef(raw['J-point'][:3072],ar)[0,1]);binding['original_A4_correlation_status']='Fixed original scalar ID-reordered only; no refit, standalone candidate or closed-route rescue'
            bf=out/'binding_manifest.json';immutable(bf,binding);artifacts.append(dict(path=str(bf),SHA=sha(bf)))
            complete=dict(UTC=utc(),complete=True,stage_ID=c['stage_ID'],stage_config_SHA=args.config_sha,producer_SHA=sha(__file__),candidate_reports=reports,artifacts=artifacts,
                frozen_before_epoch=c['cutoff_epoch'],explicit_late_refreeze_authorization=c['authorization'],late_after_original_T18=True,no_API_EVAL_submission_or_schedule=True)
            immutable(completefile,complete)
            print(json.dumps(dict(complete=True,path=str(completefile),SHA=sha(completefile),candidate_reports=reports,no_official_action=True)))
        except BaseException as e:
            if not(out/'attention.json').exists():immutable(out/'attention.json',dict(UTC=utc(),kind='engineering_attention',error_type=type(e).__name__,error=str(e),no_automatic_retry=True,partial_outputs_not_complete=True))
            raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True)
    group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--audit-only',action='store_true');group.add_argument('--execute',action='store_true')
    args=ap.parse_args();prepared=prepare(args)
    if args.audit_only:
        c,ids,splits,parent,fold,basev,axis,b3,lineage=prepared
        print(json.dumps(dict(audit_only=True,stage_config_SHA=args.config_sha,producer_SHA=sha(__file__),complete_axes=[*JUDGES,'B3_AUDIOBOX'],rows=len(ids),TRAIN_rows=3072,VALID_rows=2117,EVAL_rows=2189,PF_ID_folds_source_lineage_verified=True,no_model_fit_CV_candidate_CSV_API_or_writes=True)));return
    execute(args,prepared)


if __name__=='__main__':main()
