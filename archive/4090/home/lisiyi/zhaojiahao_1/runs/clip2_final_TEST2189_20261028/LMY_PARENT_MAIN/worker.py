"""Fixed LMY public input assembly and original CPU parent/readout; no fitting."""
import argparse, ast, copy, hashlib, json, os, pathlib, sys, time, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--phase', choices=['static', 'parent'], required=True)
    ap.add_argument('--controls-only', action='store_true')
    ap.add_argument('--fixed4090', action='store_true', help='Keep original weights and record cross-device numerical differences')
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    cfg = json.loads((root / 'protocol.json').read_text())
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verified = {}

    def sha(p):
        h = hashlib.sha256()
        with pathlib.Path(p).open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
                h.update(block)
        return h.hexdigest()

    def pin(key):
        item = cfg['assets'][key]
        p = pathlib.Path(item['path'])
        assert sha(p) == item['SHA'], key
        verified[key] = item
        return p

    def take(key, signals, names):
        with np.load(pin(key), allow_pickle=False) as z:
            ids = z['signals'].tolist()
            assert len(ids) == len(set(ids))
            ix = {s: i for i, s in enumerate(ids)}
            ii = [ix[s] for s in signals]
            return {k: z[k][ii].copy() for k in names}

    def extract(key, names, env, nested=None):
        p = pin(key)
        tree = ast.parse(p.read_text())
        if nested:
            tree = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == nested)
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), env)

    rows = json.loads(pin('selection').read_text())
    assert len(rows) == 2191
    if a.controls_only:
        rows = rows[:2]
    signals = [r['signal'] for r in rows]
    n = len(rows)
    legal = np.arange(12)[None, :] < np.array([r['n_words'] for r in rows])[:, None]
    severity = np.asarray([r['severity'] for r in rows], np.int64)
    tag = 'CONTROL' if a.controls_only else 'EVAL_WITH_CONTROLS'
    execution_contract = 'identical_frozen_weights_original_recipe_4090'
    receipts = {}
    if a.phase == 'parent' and a.fixed4090:
        assert not a.controls_only
        for key in ['native_receipt', 'listener_receipt']:
            receipts[key] = json.loads(pin(key).read_text())
            assert receipts[key]['rows'] == 2189
        nr = receipts['native_receipt']
        assert nr['execution_contract'] == execution_contract
        assert nr['original_recipe_and_parameters_unchanged']
        assert nr['base_unchanged'] and nr['adapters_unchanged']
        assert receipts['listener_receipt']['controls_passed']
    if a.phase == 'static':
        raw = take('side_raw', signals, ['confidence_v3', 'confidence_v2', 'wer',
                                       'extra_wer', 'candidate', 'paired_acoustic', 'strict'])
        lm4 = take('lm4', signals, ['lm4'])['lm4']
        values = np.zeros((n, 12, 261), np.float32)
        support = np.zeros((n, 12), bool)
        pooled = np.zeros((n, 1280), np.float32)
        for i, r in enumerate(rows):
            p = pathlib.Path(cfg['word_rows']) / (r['signal'] + '.npz')
            with np.load(p, allow_pickle=False) as z:
                assert str(z['audio_SHA']) == r['audio_SHA']
                count = r['n_words']
                values[i, :count] = z['values'].astype(np.float32)
                support[i, :count] = z['support']
                pooled[i] = z['pooled']
        assert not (support & ~legal).any()
        env = dict(np=np, SIDE=['confidence_v3', 'confidence_v2', 'wer', 'extra_wer', 'candidate', 'paired'],
                   DIMS=[12, 12, 1, 4, 1, 24])
        def require(test, message):
            assert test, message
        env['require'] = require
        extract('paired_source', ['assemble_paired'], env)
        extract('side_source', ['normalized_side'], env)
        raw['paired'] = env['assemble_paired'](raw['paired_acoustic'], raw['strict'], lm4)
        raw['pooled'] = pooled
        norm = json.loads(pin('side_norm').read_text())
        side = env['normalized_side'](raw, norm)
        count = legal.sum(1)
        position = (np.arange(12)[None, :] + .5) / count[:, None]
        inverse = np.broadcast_to(1 / count[:, None], legal.shape)
        global60 = np.concatenate([np.broadcast_to(side[:, None, :], (n, 12, 54)),
            position[..., None], inverse[..., None], support[..., None],
            np.broadcast_to(np.eye(3, dtype=np.float32)[severity][:, None, :], (n, 12, 3))], -1).astype(np.float32)
        norm = json.loads(pin('context_norm').read_text())
        inds = np.r_[np.arange(261, 315), np.arange(320, 326)]
        global60 = np.clip((global60 - np.asarray(norm['mean'])[inds]) /
                           np.asarray(norm['std'])[inds], -6, 6).astype(np.float32)
        free = take('free_asr', signals, ['normalized9', 'available'])
        f70 = np.concatenate([global60, free['normalized9'], free['available'][..., None]], -1).astype(np.float32)
        f70[~legal] = 0
        reference = take('reference_f96', signals[:2], ['normalized96'])['normalized96'][..., :70]
        gap = float(abs(f70[:2] - reference).max())
        assert gap <= 2e-5, ('f70 control', gap)
        p = root / (tag + '_STATIC.npz')
        np.savez_compressed(p, signals=np.asarray(signals), words=legal, severity=severity,
                            values=values, support=support, pooled=pooled, side=side, f70=f70)
        result = dict(rows=n, output_SHA=sha(p), f70_control_maxgap=gap, training_updates=0,
                      verified_assets=verified)
        (root / (tag + '_STATIC.json')).write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != 'verified_assets'}), flush=True)
        return

    static_path = root / (tag + '_STATIC.npz')
    if a.fixed4090:
        static_receipt = json.loads(pin('static_receipt').read_text())
        assert static_receipt['rows'] == 2191 and static_receipt['f70_control_maxgap'] <= 2e-5
        assert sha(static_path) == static_receipt['output_SHA']
    with np.load(static_path, allow_pickle=False) as z:
        static = {k: z[k].copy() for k in z.files}
    assert static['signals'].tolist() == signals
    with np.load(pin('text'), allow_pickle=False) as z:
        index = {s: i for i, s in enumerate(z['signals'].tolist())}
        tl = z['token_lengths']; wl = z['word_lengths']
        toff = np.r_[0, np.cumsum(tl)]; woff = np.r_[0, np.cumsum(wl)]
        all_tokens, all_words, all_mapped = z['tokens'], z['word_vectors'], z['mapped']
        ii = np.asarray([index[s] for s in signals])
        tmax = int(tl[ii].max())
        static['tokens'] = np.zeros((n, tmax, 768), np.float32)
        static['token_lengths'] = tl[ii].copy()
        static['word_text'] = np.zeros((n, 12, 768), np.float32)
        static['mapped'] = np.zeros((n, 12), bool)
        for j, i in enumerate(ii):
            assert wl[i] == legal[j].sum()
            static['tokens'][j, :tl[i]] = all_tokens[toff[i]:toff[i+1]]
            static['word_text'][j, :wl[i]] = all_words[woff[i]:woff[i+1]]
            static['mapped'][j, :wl[i]] = all_mapped[woff[i]:woff[i+1]]
    static['prompt_index'] = np.arange(n)
    listener = np.zeros((n, 12, 3), np.float32)
    frames = []
    raw45 = np.zeros((n, 12, 45), np.float32)
    x45 = np.zeros_like(raw45)
    row_bindings = []
    for i, r in enumerate(rows):
        lp = pathlib.Path(cfg['listener_rows']) / (r['signal'] + '.npz')
        with np.load(lp, allow_pickle=False) as z:
            assert str(z['audio_SHA']) == r['audio_SHA']
            listener[i, :r['n_words']] = z['adapted_word3']
        native_rows = cfg.get('native_diagnostic_rows', cfg['native_rows']) if a.controls_only else cfg['native_rows']
        qp = pathlib.Path(native_rows) / (r['signal'] + '.npz')
        with np.load(qp, allow_pickle=False) as z:
            assert str(z['audio_SHA']) == r['audio_SHA']
            assert str(z['text']) == r['prompt']
            ff = z['frames'].copy()
            assert ff.ndim == 2 and ff.shape[1] == 2048 and np.isfinite(ff).all()
            frames.append(torch.from_numpy(ff))
            raw45[i] = z['raw45']; x45[i] = z['x45']
        row_bindings.append(dict(signal=r['signal'], listener_SHA=sha(lp), native_SHA=sha(qp)))
        if i % 128 == 0:
            (root/'progress.json').write_text(json.dumps(dict(stage='read_inputs', done=i, total=n)))
    if a.fixed4090:
        with np.load(pin('native_features'), allow_pickle=False) as z:
            assert z['signals'].tolist() == signals[2:]
            assert np.array_equal(z['raw45'], raw45[2:]) and np.array_equal(z['x45'], x45[2:])
        (root/'input_row_manifest.json').write_text(json.dumps(row_bindings, indent=2))
    for name, source in json.loads(pin('module_sources').read_text()).items():
        module = types.ModuleType(name); sys.modules[name] = module
        exec(compile(source, '<frozen module: ' + name + '>', 'exec'), module.__dict__)
    initial = torch.load(pin('initial'), map_location='cpu', weights_only=False).eval().requires_grad_(False)
    assert tuple(initial.q_proj.weight.shape) == (192, 2048)
    heads = {}
    exec(compile(pin('strict24_head').read_text(), '<frozen original heads>', 'exec'), heads)
    parents = {}
    for seed in [44, 45]:
        stage = copy.deepcopy(initial)
        stage.load_state_dict(torch.load(pin('stage' + str(seed)), map_location='cpu', weights_only=True)['weights'])
        score = heads['FineResidual'](heads['CTCReadout'](heads['TFReadout'](), seed), seed, 'bounded_pooled')
        score.load_state_dict(torch.load(pin('beta' + str(seed)), map_location='cpu', weights_only=True)['weights'])
        parents[seed] = dict(stage=stage.eval().requires_grad_(False), score=score.eval().requires_grad_(False),
                             norm=json.loads(pin('parent_norm' + str(seed)).read_text()))
    env = dict(torch=torch, np=np, parent_models=parents, static=static, words=legal, listener_adapted=listener)
    extract('parent_source', ['parent_batch'], env, 'run')
    hp = {}
    for seed in [44, 45]:
        batches = []
        for lo in range(0, n, 64):
            batches.append(env['parent_batch'](np.arange(lo, min(lo+64, n)), frames[lo:lo+64], seed))
            progress = dict(stage='parent', seed=seed, done=min(lo+64, n), total=n)
            (root/'progress.json').write_text(json.dumps(progress))
            print(json.dumps(progress), flush=True)
        hp[seed] = np.concatenate(batches)
    stack = np.stack([hp[44], hp[45]], -1)
    logits = np.log(np.clip(stack, 1e-7, 1-1e-7)) - np.log1p(-np.clip(stack, 1e-7, 1-1e-7))
    count = legal.sum(1)
    common = np.concatenate([static['values'], np.broadcast_to(static['side'][:, None], (n, 12, 54)),
        listener, logits, ((np.arange(12)[None] + .5) / count[:, None])[..., None],
        np.broadcast_to(1/count[:, None], legal.shape)[..., None], static['support'][..., None],
        np.broadcast_to(np.eye(3, dtype=np.float32)[severity][:, None], (n, 12, 3))], -1).astype(np.float32)
    norm = json.loads(pin('x9_norm').read_text())
    common = np.clip((common - np.asarray(norm['common_mean'])) / np.asarray(norm['common_std']), -6, 6).astype(np.float32)
    common[:, :, :261][~static['support']] = 0; common[~legal] = 0
    native13 = np.clip((raw45[:, :, :13] - np.asarray(norm['native_mean'])[:, 1]) /
                      np.asarray(norm['native_std'])[:, 1], -6, 6).astype(np.float32)
    native13[~legal] = 0
    x339 = np.concatenate([common, native13], -1)
    env = dict(torch=torch, nn=nn, F=F)
    extract('x9_source', ['NativeWordScore'], env)
    result = dict(signals=np.asarray(signals), words=legal, x45=x45, f70=static['f70'])
    proof = {}
    reference = take('reference_main', signals[:2], ['parent44', 'parent45', 'x45'])
    for seed in [44, 45]:
        cp = torch.load(pin('x9_head' + str(seed)), map_location='cpu', weights_only=True)
        assert cp['normalization_sha256'] == cfg['assets']['x9_norm']['SHA']
        model = env['NativeWordScore'](seed, cp['dispersion'])
        model.load_state_dict(cp['weights']); model.eval().requires_grad_(False)
        with torch.inference_mode():
            _, wp, _ = model(torch.from_numpy(x339), torch.from_numpy(hp[seed]), torch.from_numpy(legal))
        parent = (hp[seed] + wp.numpy()) * .5
        parent[~legal] = 0
        gap = float(abs(parent[:2] - reference['parent' + str(seed)]).max())
        proof[str(seed)] = dict(parent_maxgap=gap, strict_numeric_replay_passed=gap <= 2e-6)
        if not a.controls_only and not a.fixed4090:
            assert gap <= 2e-6, proof
        xx = np.zeros((n, 12, 381), np.float32)
        xx[:, :, :45] = x45
        xx[:, :, 306:360] = static['f70'][:, :, :54]
        xx[:, :, 365:371] = static['f70'][:, :, 54:60]
        xx[:, :, 371:381] = static['f70'][:, :, 60:70]
        cp = torch.load(pin('main_head' + str(seed)), map_location='cpu', weights_only=True)
        w = {k.removeprefix('branch.'): v for k, v in cp['head_weights'].items() if k.startswith('branch.')}
        net = nn.Sequential(nn.Linear(381, 16), nn.GELU(), nn.Linear(16, 1))
        net.load_state_dict(w); net.eval().requires_grad_(False)
        with torch.inference_mode():
            z = torch.logit(torch.from_numpy(parent).clamp(1e-7, 1-1e-7)) + net(torch.from_numpy(xx)).squeeze(-1)
            wp = z.sigmoid() * torch.from_numpy(legal)
            prediction = wp.sum(1) / torch.from_numpy(legal).sum(1)
        result['human' + str(seed)] = hp[seed]
        result['parent' + str(seed)] = parent
        result['main_word' + str(seed)] = wp.numpy()
        result['main_prediction' + str(seed)] = prediction.numpy()
    p = root / (tag + '_PARENT_MAIN.npz')
    np.savez_compressed(p, **result)
    report = dict(rows=n, output_SHA=sha(p), controls=proof, verified_assets=verified,
                  training_updates=0, diagnostic_only=a.controls_only,
                  execution_contract=execution_contract if a.fixed4090 else 'strict_original_numeric_replay',
                  original_recipe_and_parameters_unchanged=True,
                  strict_numeric_replay_passed=all(v['strict_numeric_replay_passed'] for v in proof.values()),
                  input_row_manifest_SHA=sha(root/'input_row_manifest.json') if a.fixed4090 else None,
                  input_completion_receipts=receipts)
    (root / (tag + '_PARENT_MAIN.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'verified_assets'}), flush=True)


if __name__ == '__main__':
    main()
