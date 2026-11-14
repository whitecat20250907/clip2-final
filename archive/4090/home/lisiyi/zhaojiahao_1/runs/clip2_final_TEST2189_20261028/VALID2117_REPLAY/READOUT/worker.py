"""Apply the frozen LMY arithmetic to all VALID2117 with fresh Qwen-dependent inputs."""
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
    ap.add_argument('--mode', choices=['qualify', 'replay'], required=True)
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

    parent_receipt=json.loads(pinned(cfg['replay_parent_receipt']).read_text())
    assert parent_receipt['rows']==2117 and parent_receipt['training_updates']==parent_receipt['model_changes']==0
    fresh=nz(cfg['replay_parent']);ids=fresh['signals'].tolist();assert ids==valid_ids
    cache=env['cache']
    cache['main_cache']={k:fresh[k]for k in ['signals','words','x45','parent44','parent45']}
    cache['main_confidence96']=dict(signals=fresh['signals'],normalized96=fresh['f70'])
    for seed in [44,45]:
        ss=str(seed);cache['main_sealed'+ss]=dict(signals=fresh['signals'],words=fresh['words'],prediction=fresh['main_prediction'+ss],word_probability=fresh['main_word'+ss])
    vd.update(x45=fresh['x45'],f70=fresh['f70']);cache['valid_base920']=build920(vd)
    new,_,_=env['frozen_forward'](ids,'VALID_FULL_REPLAY')
    old=np.asarray([float(r[1])/100 for r in expected]);delta=100*(new-old)
    assert len(delta)==2117 and np.isfinite(delta).all()
    p=root/'VALID_LMY_REPLAY.npz';np.savez_compressed(p,signal_ids=np.asarray(ids),old_prediction=old,prediction=new,difference_points=delta)
    report=dict(rows=2117,output_SHA=sha(p),max_abs_difference_points=float(abs(delta).max()),prediction_RMS_difference_points=float(np.sqrt(np.mean(delta**2))),mean_signed_difference_points=float(delta.mean()),absolute_difference_quantiles={str(q):float(np.quantile(abs(delta),q))for q in [.5,.9,.95,.99,1.]},training_updates=0,model_changes=0,labels_read=False,scope='Full VALID2117; fresh Qwen17 raw45/audio frames and recomputed downstream parent, LMY and joint; other fixed inputs reused',original_VALID_CSV_identity=m['expected_VALID_CSV_SHA'],original_cached_VALID_byte_identical=exact,parent_receipt_SHA=sha(pinned(cfg['replay_parent_receipt'])),bindings=cfg)
    (root/'valid_replay_complete.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='bindings'}),flush=True)

if __name__=='__main__':main()
