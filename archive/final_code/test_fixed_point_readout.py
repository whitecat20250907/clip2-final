"""Predict-only use of the four original positive scales and TRAIN norms."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fixed_increment(raw,fit):
    raw=np.asarray(raw,np.float64)
    assert raw.ndim==2 and raw.shape[1]==6 and np.isfinite(raw).all()
    q=(raw-np.asarray(fit['question_TRAIN_means']))/np.maximum(np.asarray(fit['question_TRAIN_SD_population']),1e-6)
    scalar=np.clip((q.mean(1)-fit['scalar_TRAIN_mean'])/max(fit['scalar_TRAIN_SD_population'],1e-6),-5,5)
    assert fit['positive_b']>=0
    return scalar*fit['positive_b']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);a=ap.parse_args();b=Path(a.bundle).resolve();v=b.parents[2];j=v/'next_jury_20261024'
    contract=json.loads((b/'replay_contract.json').read_text());assets=json.loads((b/'original_assets/ASSET_INDEX.json').read_text())
    p=v/'third_round_20261023/Z13_attribution_20261023/PRIOR_FIXED/candidate/valid_predictions.csv'
    assert sha(p)=='a32b857ca9bf1da0bb4dcdcdf215bd4034ed74a7bfa060865b100d35b2209b74'
    rows=list(csv.reader(p.open(newline='')));ids=[r[0]for r in rows];parent=np.asarray([float(r[1])for r in rows]);assert len(set(ids))==2117
    sources={
      'MF':(j/'broad_official_trials_20261026/collected_exports/MF/raw_fields.npz',j/'broad_official_trials_20261026/readouts/MF/fit_result.json','raw_prompt_fields'),
      'Phi':(j/'broad_official_trials_20261026/collected_exports/Phi4MM/raw_fields.npz',j/'broad_official_trials_20261026/readouts/Phi4MM/fit_result.json','raw_prompt_fields'),
      'MOSS':(j/'raw_axes_evidence/MOSS/production/MOSS_AXES/all7378_raw_axes.npz',j/'per_model_diagnostics_20261026/MOSS/fit_result.json','ear_mean_oriented'),
      'Omni':(j/'continuation_20261026_084003/raw_exports/Qwen25Omni7B/raw_fields.npz',j/'per_model_diagnostics_20261026/Qwen25Omni7B/fit_result.json','raw_prompt_fields')}
    fixed_b={'MF':.24819814829422285,'Phi':.3224447682769368,'MOSS':.39373701982067894,'Omni':.17250570338885765}
    eid=[r['signal_id']for r in json.loads((v/'EVAL_raw_readiness_1405/eval_public_manifest.json').read_text())['rows']];proof={};out={}
    for name,(rawfile,fitfile,key)in sources.items():
        fit=json.loads(fitfile.read_text());assert fit['positive_b']==fixed_b[name]
        with np.load(rawfile,allow_pickle=False)as z:
            ss=z['signal_ids'].tolist();raw=z[key].copy()
        assert len(ss)==len(set(ss));lookup={s:i for i,s in enumerate(ss)}
        increment=fixed_increment(raw,fit);pred=np.clip(parent+increment[[lookup[s]for s in ids]],0,100)
        ref=contract['files'][name+'_VALID'];refp=b/ref['relative_path'];assert sha(refp)==ref['SHA']
        rr=list(csv.reader(refp.open(newline='')));assert [r[0]for r in rr]==ids
        gap=float(np.abs(pred-np.asarray([float(r[1])for r in rr])).max());assert gap<=5.1e-11,(name,gap)
        proof[name]=dict(raw_path=str(rawfile),raw_SHA=sha(rawfile),fit_path=str(fitfile),fit_SHA=sha(fitfile),fixed_b=fit['positive_b'],VALID_maxgap_points=gap,reference_CSV_SHA=ref['SHA'])
        if set(eid)<=set(lookup):out[name+'_increment_points']=increment[[lookup[s]for s in eid]]
    assert set(out)=={'MOSS_increment_points','Omni_increment_points'}
    dest=b/'outputs/TEST_components/MOSS_Omni_fixed_increments.npz'
    with dest.open('xb')as f:np.savez_compressed(f,signal_ids=np.asarray(eid),**out)
    receipt=dict(cases=proof,TRAIN_normalization_reused=True,training_updates=0,VALID_labels_read=False,EVAL_rows=2189,output_SHA=sha(dest),output_is_final_prediction=False)
    (b/'outputs/TEST_POINT_READOUT_QUALIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':main()
