"""Export the frozen joint model on completed EVAL inputs; never train or submit."""
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
    out = b / 'outputs/CSV_DELIVERY'
    out.mkdir(exist_ok=True)
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
    lroot = b/'outputs/TEST_LMY_FINAL_READOUT/collected'
    completion = json.loads((lroot/'complete.json').read_text())
    assert completion['rows'] == 2189 and completion['training_updates'] == completion['model_changes'] == 0
    assert completion['execution_contract'] == 'identical_frozen_weights_original_recipe_4090'
    lmyfile = lroot/'LMY_EVAL_FIXED.npz'
    assert sha(lmyfile) == completion['output_SHA']
    with np.load(lmyfile, allow_pickle=False) as z:
        ids, raw_lmy = z['signal_ids'].copy(), z['prediction'].copy()
    assert len(ids) == len(set(ids.tolist())) == 2189
    # Match the original LMY interface: 12 decimal places in percentage units.
    lmybytes = payload(ids, raw_lmy*100, digits=12)
    lmyrows = list(csv.reader(io.StringIO(lmybytes.decode())))[1:]
    lmy = np.asarray([float(r[1])/100 for r in lmyrows])
    ownfile = out/'OWN_EVAL2189.csv'
    assert sha(ownfile) == 'ec2a9597b36d6ea4578b5171ed5d5decaf58bce1321307f5677e3a077490f0c9'
    ownrows = list(csv.reader(ownfile.open(newline='')))
    assert ownrows.pop(0) == ['signal_id', 'intelligibility_score']
    assert [r[0] for r in ownrows] == ids.tolist()
    own = np.asarray([float(r[1])/100 for r in ownrows])
    proof = json.loads((b/'outputs/TEST_PCA8_qualification.json').read_text())
    assert proof['EVAL_PCA8_recomputed_equal'] and proof['new_PCA_fits'] == proof['new_normalization_fits'] == 0
    ep = next(Path(p) for p in proof['source_files'] if p.endswith('final_frozen_prediction\\environment.npz'))
    assert sha(ep) == '180cb38a10501f8c1fa26b8ca410e83108cff6524a53968333c479de986d1358'
    with np.load(ep, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ['signal_ids', 'audio8', 'severity', 'counts']}
    assert data['signal_ids'].tolist() == ids.tolist()
    selection = json.loads((lroot/'selection.json').read_text())[2:]
    assert [r['signal'] for r in selection] == ids.tolist()
    assert np.array_equal(data['severity'], [r['severity'] for r in selection])
    assert np.array_equal(data['counts'], [r['n_words'] for r in selection])
    w0 = env['W0']
    data.update(own=own, LMY=lmy, J0=w0*own+(1-w0)*lmy)
    joint, delta = env['predict'](net, data, saved['normalization'])
    _, ood = env['transform'](env['metadata'](data), saved['normalization'])
    assert np.count_nonzero(delta[ood]) == 0
    assert np.max(abs(joint-((w0+delta)*own+(1-w0-delta)*lmy))) < 3e-16
    files = {}
    for name, body in [('LMY_EVAL2189_FIXED4090.csv', lmybytes),
                       ('JOINT_EVAL2189_FIXED4090.csv', payload(ids, joint*100))]:
        path = out/name
        if path.exists():
            assert path.read_bytes() == body
        else:
            path.write_bytes(body)
        records = list(csv.reader(path.open(newline='')))
        assert records.pop(0) == ['signal_id', 'intelligibility_score']
        assert [r[0] for r in records] == ids.tolist()
        values = np.asarray([float(r[1]) for r in records])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 100)).all()
        files[name] = dict(rows=2189, SHA=sha(path), range_points=[float(values.min()), float(values.max())])
    np.savez_compressed(out/'JOINT_EVAL_DIAGNOSTICS.npz', signal_ids=ids, own=own, LMY=lmy,
                        delta=delta, original_weight=w0, OOD=ood, prediction=joint)
    receipt = dict(files=files, parameters=13, training_updates=0, model_changes=0,
        original_weight=w0, actual_own_weight_range=[float((w0+delta).min()), float((w0+delta).max())],
        OOD_rows=int(ood.sum()), original_VALID_byte_identical=True,
        strict_5090_numeric_replay_passed=completion['strict_numeric_replay_passed'],
        execution_contract=completion['execution_contract'],
        LMY_completion_SHA=sha(lroot/'complete.json'), LMY_NPZ_SHA=sha(lmyfile),
        selection_SHA=sha(lroot/'selection.json'), environment_SHA=sha(ep),
        adapter_model_SHA=sha(model), adapter_source_SHA=sha(src), export_source_SHA=sha(__file__),
        EVAL_labels_read=False, platform_submissions=0, scheduled_tasks_created=0)
    (out/'JOINT_EVAL_COMPLETE.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
