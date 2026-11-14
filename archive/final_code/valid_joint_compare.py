"""Compare original and replayed frozen joint VALID2117 predictions without labels."""
import argparse, ast, csv, hashlib, io, json
from pathlib import Path
import numpy as np
import torch


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def payload(ids, values, digits=10, header=True):
    f = io.StringIO(newline='')
    w = csv.writer(f, lineterminator='\r\n')
    if header:
        w.writerow(['signal_id', 'intelligibility_score'])
    w.writerows(zip(ids, [format(float(v), '.' + str(digits) + 'f') for v in values]))
    return f.getvalue().encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qualify-only', action='store_true')
    a = ap.parse_args()
    b = Path(__file__).resolve().parent
    out = b / 'outputs/VALID2117_REPLAY/results'
    out.mkdir(parents=True, exist_ok=True)
    src = b / 'scientific_sources/joint13_train.py'
    assert sha(src) == 'c77d1cf21e4e0f38adc19f5d45c913c45d59b32aee95692986cd609ea756e494'
    model = b / 'replay_inputs/ADAPTER_MODEL.pt'
    assert sha(model) == '444a577b1fd987867e1eda464c2053ca2045f1a9b670595f1de57a553686db89'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    names = {'metadata', 'transform', 'Adapter', 'predict'}
    nodes = [n for n in ast.parse(src.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    def require(condition, message):
        assert condition, message
    env = dict(np=np, torch=torch, W0=.5609719773532506, require=require)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src), 'exec'), env)
    saved = torch.load(model, map_location='cpu', weights_only=True)
    assert saved['fold'] == -1 and saved['parameters'] == 13
    assert (np.asarray(saved['normalization']['std']) >= .01).all()
    net = env['Adapter']().eval().requires_grad_(False)
    net.load_state_dict(saved['state'], strict=True)
    assert sum(p.numel() for p in net.parameters()) == 13
    contract = json.loads((b / 'replay_contract.json').read_text())
    for key, r in contract['files'].items():
        assert sha(b/r['relative_path']) == r['SHA'], key
    with np.load(b/'replay_inputs/VALID_DATA.npz', allow_pickle=False) as z:
        vd = {k: z[k].copy() for k in ['signal_ids', 'audio8', 'severity', 'counts', 'own', 'LMY']}
    rows = list(csv.reader((b/'replay_inputs/ZERO_JOINT_VALID.csv').open(newline='')))
    assert [r[0] for r in rows] == vd['signal_ids'].tolist()
    vd['J0'] = np.asarray([float(r[1])/100 for r in rows])
    vp, _ = env['predict'](net, vd, saved['normalization'])
    assert hashlib.sha256(payload(vd['signal_ids'], vp*100, header=False)).hexdigest() == contract['files']['FINAL_JOINT_VALID']['SHA']
    qualification = dict(VALID_byte_identical=True, rows=2117, parameters=13,
                         model_SHA=sha(model), source_SHA=sha(src), fits=0)
    (out/'JOINT_READOUT_QUALIFICATION.json').write_text(json.dumps(qualification, indent=2))
    print(json.dumps(qualification), flush=True)
    if a.qualify_only:
        return
    lroot=b/'outputs/VALID2117_REPLAY/collected'
    receipt=json.loads((lroot/'valid_replay_complete.json').read_text());assert receipt['rows']==2117 and receipt['training_updates']==receipt['model_changes']==0
    lp=lroot/'VALID_LMY_REPLAY.npz';assert sha(lp)==receipt['output_SHA']
    with np.load(lp,allow_pickle=False)as z:
        ids=z['signal_ids'].copy();fresh=z['prediction'].copy();assert ids.tolist()==vd['signal_ids'].tolist()
    new_lmy=np.asarray([float(format(float(x*100),'.12f'))/100 for x in fresh])
    old_lmy=vd['LMY'].copy();old_own=vd['own'].copy();w0=env['W0']
    # Preserve the canonical ten-decimal zero-joint CSV boundary.
    candidate_zero=w0*old_own+(1-w0)*new_lmy
    vd['J0']=np.asarray([float(format(float(v*100),'.10f'))/100 for v in candidate_zero]);vd['LMY']=new_lmy
    new_joint,delta=env['predict'](net,vd,saved['normalization'])
    body=payload(ids,new_joint*100);(out/'JOINT_VALID2117_REPLAY4090.csv').write_bytes(body)
    (out/'LMY_VALID2117_REPLAY4090.csv').write_bytes(payload(ids,new_lmy*100,digits=12))
    old_rows=list(csv.reader((b/'replay_inputs/FINAL_JOINT_VALID.csv').open(newline='')))
    assert [r[0]for r in old_rows]==ids.tolist();old_points=np.asarray([float(r[1])for r in old_rows])
    new_rows=list(csv.reader(io.StringIO(body.decode())))[1:];new_points=np.asarray([float(r[1])for r in new_rows]);diff=new_points-old_points
    assert np.isfinite(new_points).all()and((new_points>=0)&(new_points<=100)).all()
    pred_rms=float(np.sqrt(np.mean(diff**2)));bound=float(np.nextafter(pred_rms+1e-10,np.inf));official=15.294557340680246
    lmy_diff=100*(new_lmy-old_lmy)
    ideal=(1-w0-delta)*lmy_diff
    assert np.max(abs(diff-ideal))<2e-9
    ix=int(np.argmax(abs(diff)))
    metrics=dict(rows=2117,final_prediction_RMS_difference_points=pred_rms,RMSE_change_upper_bound_points=bound,official_original_RMSE=official,guaranteed_new_RMSE_interval_on_same_VALID=[max(0.,official-bound),official+bound],actual_new_RMSE_measured=False,labels_read=False,mean_abs_difference_points=float(abs(diff).mean()),max_abs_difference_points=float(abs(diff).max()),max_difference_signal=str(ids[ix]),mean_signed_difference_points=float(diff.mean()),absolute_difference_quantiles={str(q):float(np.quantile(abs(diff),q))for q in [.5,.9,.95,.99,1.]},counts_abs_difference_over={str(v):int((abs(diff)>v).sum())for v in [.001,.01,.05,.1,.2,.5,1.]},LMY_prediction_RMS_difference_points=float(np.sqrt(np.mean(lmy_diff**2))),LMY_max_abs_difference_points=float(abs(lmy_diff).max()),training_updates=0,model_changes=0,fusion_changes=0,platform_submissions=0,scope=receipt['scope'],bound_applies_to='This VALID2117 cohort only; not a guarantee for EVAL2189',bound_derivation='reverse triangle inequality: abs(RMSE(new,y)-RMSE(old,y)) <= norm(new-old,2)/sqrt(N)',conservative_roundoff_margin_points=1e-10,original_VALID_source_SHA=sha(b/'replay_inputs/FINAL_JOINT_VALID.csv'),new_VALID_CSV_SHA=sha(out/'JOINT_VALID2117_REPLAY4090.csv'),LMY_readout_completion_SHA=sha(lroot/'valid_replay_complete.json'),LMY_output_SHA=sha(lp),adapter_checkpoint_SHA=sha(model),source_SHA=sha(__file__))
    with (out/'VALID2117_PREDICTION_COMPARISON.csv').open('w',newline='')as f:
        w=csv.writer(f);w.writerow(['signal_id','original_joint','replay4090_joint','difference_points','original_LMY','replay4090_LMY','LMY_difference_points','LMY_weight'])
        w.writerows(zip(ids,old_points,new_points,diff,old_lmy*100,new_lmy*100,lmy_diff,1-w0-delta))
    np.savez_compressed(out/'VALID2117_PREDICTION_COMPARISON.npz',signal_ids=ids,original_joint=old_points,replay_joint=new_points,difference_points=diff,LMY_difference_points=lmy_diff,delta=delta)
    (out/'FULL_VALID_COMPARISON.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
