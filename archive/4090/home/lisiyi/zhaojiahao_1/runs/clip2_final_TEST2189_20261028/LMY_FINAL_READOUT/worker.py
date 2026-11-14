"""Original frozen LMY arithmetic with an explicitly bound EVAL input adapter."""
import argparse, ast, csv, hashlib, io, itertools, json, os, pathlib, sys


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--mode', choices=['qualify', 'diagnose-controls', 'eval'], required=True)
    ap.add_argument('--fixed4090', action='store_true', help='Explicit same-weight cross-device execution contract')
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    cfg = json.loads((root/'protocol.json').read_text())
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import torch
    from torch import nn
    from scipy.special import expit, logit
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    def pinned(item):
        p = pathlib.Path(item['path'])
        assert sha(p) == item['SHA'], str(p)
        return p

    original = pinned(cfg['original_source'])
    assert sha(original) == '06c06f00e881d73ed9aa9b2582d2691893dca092e92d1a1ebc990e3cb1213865'
    m = json.loads(pinned(cfg['original_manifest']).read_text())
    tree = ast.parse(original.read_text())
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    names = {'pin', 'load', 'take', 'q39_train', 'standard', 'rr', 'wordnet', 'pool', 'frozen_forward'}
    nodes = [n for n in main_node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    env = dict(np=np, torch=torch, nn=nn, expit=expit, logit=logit, pathlib=pathlib,
               m=m, sha=sha, verified={}, cache={}, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(original), 'exec'), env)
    valid_ids = m['VALID_signal_ids']
    qv, gfv, stages = env['frozen_forward'](valid_ids, 'VALID')
    expected = list(csv.reader(env['pin']('valid_csv').open(newline='')))
    assert [r[0] for r in expected] == valid_ids
    gap = float(abs(qv-np.asarray([float(r[1])/100 for r in expected])).max())
    assert gap <= m['maximum_replay_gap_native']
    f = io.StringIO(newline='')
    csv.writer(f, lineterminator='\n').writerows(zip(valid_ids, [format(float(v*100), '.12f') for v in qv]))
    exact = hashlib.sha256(f.getvalue().encode()).hexdigest() == m['expected_VALID_CSV_SHA']
    assert exact, 'Frozen VALID export must remain byte-identical'

    def nz(item):
        with np.load(pinned(item), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    support_norm = nz(cfg['support_norm'])
    joint_norm = nz(cfg['joint_norm'])
    source = pinned(cfg['sameword_source'])
    funcs = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'make_word_extra']
    assert len(funcs) == 1
    groups = [0, 0, 1, 1, 2, 2]
    pairs = [(i, j) for i, j in itertools.combinations(range(6), 2) if groups[i] != groups[j]]

    def build920(data):
        legal, local, x45 = data['legal'], data['local81'], data['x45']
        support = np.stack([x45[..., 1], x45[..., 9], data['new32'][..., 0],
                            data['new32'][..., 1], local[..., 70], local[..., 71]], -1).astype(np.float64)
        missing = local[..., 80] > .5
        support[..., 4:] = np.where(missing[..., None], np.nan, support[..., 4:])
        ee = dict(np=np, support=support, legal=legal, missing=missing, pairs=pairs)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), 'exec'), ee)
        sw = ee['make_word_extra'](support_norm)['sameword_full49']
        band = data['band24']; sev = data['severity']
        xx = np.c_[sw, band]
        z = np.clip((np.where(np.isfinite(xx), xx, joint_norm['median'][573:646]) -
                     joint_norm['mean'][573:646]) / joint_norm['std'][573:646], -6, 6)
        cod = np.c_[(sev == 0).astype(float)-(sev == 2), (sev == 1).astype(float)-(sev == 2)]
        extra = np.c_[(z[:, :49, None]*cod[:, None, :]).reshape(len(xx), 98),
                      (z[:, 49:, None]*cod[:, None, :]).reshape(len(xx), 48)]
        pool64 = env['pool'](np.concatenate([data['old32'], data['new32']], -1), legal)
        pool115 = env['pool'](np.concatenate([x45, data['f70']], -1), legal)
        result = dict(signals=data['signals'], legal=legal)
        for seed in [44, 45]:
            base = np.c_[data['q39_'+str(seed)], data['native48'], pool64, pool115]
            assert base.shape == (len(legal), 445)
            x = np.c_[base, data['h14_RP128'], sw, band, data['audio2_abs128'], extra]
            assert x.shape == (len(legal), 920)
            result['base'+str(seed)] = x
        return result

    take = env['take']
    vd = take('valid_native81', valid_ids, ['local81', 'legal', 'native48'])
    vd.update(take('valid_tdt', valid_ids, ['old32', 'new32']))
    vd.update(take('main_cache', valid_ids, ['x45']))
    vd['f70'] = take('main_confidence96', valid_ids, ['normalized96'])['normalized96'][..., :70]
    vf = nz(cfg['valid_prefix'])
    assert vf['signals'].tolist() == valid_ids
    vd.update({k: vf[k] for k in ['h14_RP128', 'band24', 'audio2_abs128']})
    vm = json.loads(pinned(cfg['valid_metadata']).read_text())
    assert [r['signal'] for r in vm] == valid_ids
    vd['severity'] = np.asarray([{'No Loss': 0, 'Mild': 1, 'Moderate': 2}[r['hearing_loss']] for r in vm])
    vd['signals'] = np.asarray(valid_ids)
    for seed in [44, 45]:
        vd['q39_'+str(seed)] = take('q39_valid'+str(seed), valid_ids, ['features'])['features']
    rebuilt = build920(vd)
    reference = take('valid_base920', valid_ids, ['base44', 'base45'])
    base_gaps = {str(s): float(np.nanmax(abs(rebuilt['base'+str(s)]-reference['base'+str(s)]))) for s in [44, 45]}
    assert max(base_gaps.values()) <= 1e-10, base_gaps
    qualification = dict(VALID_rows=2117, VALID_maxgap_native=gap, VALID_CSV_byte_identical=exact,
                         base920_reconstruction_maxgaps=base_gaps, training_updates=0,
                         original_source_SHA=sha(original), original_manifest_SHA=sha(pinned(cfg['original_manifest'])))
    (root/'READOUT_QUALIFICATION.json').write_text(json.dumps(qualification, indent=2))
    print(json.dumps(qualification), flush=True)
    if a.mode == 'qualify':
        return

    if a.mode == 'diagnose-controls':
        fresh = nz(cfg['control_parent'])
        control_ids = fresh['signals'].tolist()
        assert len(control_ids) == 2 and control_ids == valid_ids[:2]
        cache = env['cache']
        cache['main_cache'] = {k: fresh[k] for k in ['signals', 'words', 'x45', 'parent44', 'parent45']}
        cache['main_confidence96'] = dict(signals=fresh['signals'], normalized96=fresh['f70'])
        for seed in [44, 45]:
            ss = str(seed)
            cache['main_sealed'+ss] = dict(signals=fresh['signals'], words=fresh['words'],
                prediction=fresh['main_prediction'+ss], word_probability=fresh['main_word'+ss])
        data = {k: v[:2].copy() for k, v in vd.items()}
        data.update(x45=fresh['x45'], f70=fresh['f70'])
        cache['valid_base920'] = build920(data)
        new, _, _ = env['frozen_forward'](control_ids, 'NUMERICAL_CONTROL')
        delta = 100 * (new-qv[:2])
        report = dict(signals=control_ids, cached_probability=qv[:2].tolist(),
                      fresh_4090_probability=new.tolist(), difference_0to100_points=delta.tolist(),
                      max_abs_difference_points=float(abs(delta).max()),
                      diagnostic_only=True, EVAL_exported=False, training_updates=0,
                      only_fresh_Qwen17_dependent_inputs_replaced=True)
        (root/'QWEN17_FULL_CHAIN_CONTROL_DIAGNOSTIC.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return

    # Production data have their own completion receipts. Diagnostic controls
    # cannot satisfy these conditions or silently become an EVAL input.
    native_receipt = json.loads(pinned(cfg['native_receipt']).read_text())
    assert native_receipt['rows'] == 2189 and not native_receipt.get('diagnostic_only', False)
    parent_receipt = json.loads(pinned(cfg['parent_receipt']).read_text())
    assert not parent_receipt['diagnostic_only'] and parent_receipt['rows'] == 2191
    execution_contract = 'identical_frozen_weights_original_recipe_4090'
    if a.fixed4090:
        for receipt in [native_receipt, parent_receipt]:
            assert receipt['execution_contract'] == execution_contract
            assert receipt['original_recipe_and_parameters_unchanged']
        assert native_receipt['base_unchanged'] and native_receipt['adapters_unchanged']
    else:
        assert all(x['normalized45_maxgap'] <= 2e-5 for x in native_receipt['VALID_controls'])
        assert parent_receipt['strict_numeric_replay_passed']
    parent = nz(cfg['parent_main'])
    ids = parent['signals'][2:].tolist()
    assert len(ids) == len(set(ids)) == 2189
    par = {k: v[2:] for k, v in parent.items()}
    data = nz(cfg['native81'])
    assert data['signals'].tolist() == ids
    tdt = nz(cfg['tdt']); assert tdt['signals'].tolist() == ids
    data.update(tdt)
    prefix = nz(cfg['prefix']); assert prefix['signals'].tolist() == ids
    data.update(prefix)
    data.update(x45=par['x45'], f70=par['f70'])
    sel = json.loads(pinned(cfg['selection']).read_text())[2:]
    assert [r['signal'] for r in sel] == ids
    data['severity'] = np.asarray([r['severity'] for r in sel], np.int64)
    cache = env['cache']
    cache['valid_native81'] = {k: data[k] for k in ['signals', 'local81', 'native48', 'legal']}
    cache['valid_tdt'] = tdt
    cache['main_cache'] = dict(signals=par['signals'], words=par['words'], x45=par['x45'],
                               parent44=par['parent44'], parent45=par['parent45'])
    # The unchanged consumer selects only [:70]. No missing feature is padded
    # or invented to fill the legacy field name.
    cache['main_confidence96'] = dict(signals=par['signals'], normalized96=par['f70'])
    for seed in [44, 45]:
        ss = str(seed)
        q = nz(cfg['qwen'+ss]); assert q['signals'].tolist() == ids
        data['q39_'+ss] = q['q39']
        cache['q39_valid'+ss] = dict(signals=q['signals'], features=q['q39'])
        cache['stereo_valid'+ss] = q
        for family in ['PARAKEET', 'GRANITE']:
            p = nz(cfg[family+ss]); assert p['signals'].tolist() == ids
            cache['basis_valid_'+family+ss] = p
        cache['main_sealed'+ss] = dict(signals=par['signals'], words=par['words'],
            prediction=par['main_prediction'+ss], word_probability=par['main_word'+ss])
    cache['valid_base920'] = build920(data)
    mvdr = nz(cfg['mvdr']); assert mvdr['signals'].tolist() == ids
    mvdr['legal'] = data['legal']; cache['valid_mvdr'] = mvdr
    q, gf, _ = env['frozen_forward'](ids, 'EVAL')
    assert q.shape == (2189,) and np.isfinite(q).all() and q.min() >= 0 and q.max() <= 1
    p = root/'LMY_EVAL_FIXED.npz'
    np.savez_compressed(p, signal_ids=np.asarray(ids), prediction=q, GF44=gf[44], GF45=gf[45])
    result = dict(rows=2189, output_SHA=sha(p), training_updates=0, model_changes=0,
                  VALID_CSV_identity=m['expected_VALID_CSV_SHA'], input_feature_dimensions_unchanged=True,
                  legacy_normalized96_consumer_reads_only_first70=True,
                  execution_contract=execution_contract if a.fixed4090 else 'strict_original_numeric_replay',
                  strict_numeric_replay_passed=bool(native_receipt['strict_numeric_replay_passed'] and parent_receipt['strict_numeric_replay_passed']),
                  native_controls=native_receipt['VALID_controls'], parent_controls=parent_receipt['controls'],
                  input_bindings=cfg)
    (root/'complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
