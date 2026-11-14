"""Recompute the Qwen-dependent frozen parent and MAIN heads for all VALID2117."""
import argparse, ast, copy, hashlib, json, os, pathlib, sys, types

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--cached-check',action='store_true');a=ap.parse_args()
    root=pathlib.Path(a.root);cfg=json.loads((root/'parent_protocol.json').read_text())
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    verified={}
    def sha(p):
        h=hashlib.sha256()
        with pathlib.Path(p).open('rb')as f:
            for v in iter(lambda:f.read(4*1024**2),b''):h.update(v)
        return h.hexdigest()
    def pin(key):
        v=cfg['assets'][key];p=pathlib.Path(v['path']);assert sha(p)==v['SHA'],key;verified[key]=v;return p
    def take(key,signals,names):
        with np.load(pin(key),allow_pickle=False)as z:
            ids=z['signals'].tolist();assert len(ids)==len(set(ids));ix={s:i for i,s in enumerate(ids)};ii=[ix[s]for s in signals]
            return {k:z[k][ii].copy()for k in names}
    def extract(key,names,env,nested=None):
        p=pin(key);tree=ast.parse(p.read_text())
        if nested:tree=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==nested)
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names];assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
    rows=json.loads(pin('selection').read_text());signals=[r['signal']for r in rows];n=len(rows);assert n==2117
    if not a.cached_check:
        receipt=json.loads(pin('native_receipt').read_text());assert receipt['rows']==n and receipt['base_unchanged']and receipt['adapters_unchanged']
    with np.load(pin('public_static'),allow_pickle=False)as z:
        assert z['signals'].tolist()==signals
        fields=['pooled','side','values','support','words','listener','severity','tokens','token_lengths','word_text','mapped','prompt_index']
        static={k:z[k].copy()for k in fields}
    for k in ['values','support','words','listener','word_text','mapped']:
        v=static[k];assert v.shape[1]==11;static[k]=np.pad(v,[(0,0),(0,1)]+[(0,0)]*(v.ndim-2))
    legal=static['words'];severity=static['severity'];listener=static['listener']
    assert np.array_equal(legal,np.arange(12)[None,:]<np.asarray([r['n_words']for r in rows])[:,None])
    assert np.array_equal(severity,[r['severity']for r in rows])
    static['f70']=take('reference_f96',signals,['normalized96'])['normalized96'][...,:70]
    row_bindings=[]
    if a.cached_check:
        x45=take('reference_main',signals,['x45'])['x45'];raw45=np.zeros((n,12,45),np.float32)
        raw45[...,:13]=take('reference_raw13',signals,['raw13'])['raw13'];frames=[]
    else:
        native=take('native_features',signals,['raw45','x45']);raw45=native['raw45'];x45=native['x45'];frames=[];row_bindings=[]
        for i,row in enumerate(rows):
            p=pathlib.Path(cfg['native_rows'])/(row['signal']+'.npz')
            with np.load(p,allow_pickle=False)as z:
                assert str(z['audio_SHA'])==row['audio_SHA']and str(z['text'])==row['prompt']
                assert np.array_equal(z['raw45'],raw45[i])and np.array_equal(z['x45'],x45[i]);ff=z['frames'].copy();assert ff.ndim==2 and ff.shape[1]==2048 and np.isfinite(ff).all();frames.append(torch.from_numpy(ff))
            row_bindings.append(dict(signal=row['signal'],SHA=sha(p)))
            if i%128==0:(root/'parent_progress.json').write_text(json.dumps(dict(stage='read_inputs',done=i,total=n)))
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
    hp={}
    if a.cached_check:
        for seed in [44,45]:
            z=take('human_reference'+str(seed),signals,['word_probabilities'])['word_probabilities']
            hp[seed]=np.pad(z,((0,0),(0,12-z.shape[1])))*legal
    else:
        hp = {}
        for seed in [44, 45]:
            batches = []
            for lo in range(0, n, 64):
                batches.append(env['parent_batch'](np.arange(lo, min(lo+64, n)), frames[lo:lo+64], seed))
                progress = dict(stage='parent', seed=seed, done=min(lo+64, n), total=n)
                (root/'parent_progress.json').write_text(json.dumps(progress))
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
    reference = take('reference_main', signals, ['parent44', 'parent45', 'x45'])
    for seed in [44, 45]:
        cp = torch.load(pin('x9_head' + str(seed)), map_location='cpu', weights_only=True)
        assert cp['normalization_sha256'] == cfg['assets']['x9_norm']['SHA']
        model = env['NativeWordScore'](seed, cp['dispersion'])
        model.load_state_dict(cp['weights']); model.eval().requires_grad_(False)
        with torch.inference_mode():
            _, wp, _ = model(torch.from_numpy(x339), torch.from_numpy(hp[seed]), torch.from_numpy(legal))
        parent = (hp[seed] + wp.numpy()) * .5
        parent[~legal] = 0
        difference = parent.astype(np.float64)-reference['parent'+str(seed)]
        gap = float(abs(difference).max())
        proof[str(seed)] = dict(parent_maxgap=gap, parent_RMS=float(np.sqrt(np.mean(difference[legal]**2))), strict_numeric_replay_passed=gap <= 2e-6)
        if a.cached_check:assert gap<=2e-6,proof
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
        if a.cached_check:
            golden=take('main_reference'+str(seed),signals,['word_probability','prediction'])
            main_gap=float(abs(wp.numpy()-golden['word_probability']).max());proof[str(seed)]['MAIN_word_maxgap']=main_gap;assert main_gap<=2e-6,proof
        result['human' + str(seed)] = hp[seed]
        result['parent' + str(seed)] = parent
        result['main_word' + str(seed)] = wp.numpy()
        result['main_prediction' + str(seed)] = prediction.numpy()
    p=root/('CACHED_VALID_PARENT_MAIN.npz'if a.cached_check else 'VALID_PARENT_MAIN.npz');np.savez_compressed(p,**result)
    row_manifest=root/('cached_parent_row_manifest.json'if a.cached_check else 'parent_row_manifest.json');row_manifest.write_text(json.dumps(row_bindings,indent=2))
    report=dict(rows=n,output_SHA=sha(p),comparisons=proof,verified_assets=verified,training_updates=0,model_changes=0,execution_contract='identical_frozen_weights_original_recipe_4090',scope=('Full VALID2117 cached-input arithmetic qualification; original human parents, raw13, x45 and all other frozen inputs reused' if a.cached_check else 'Full VALID2117 with fresh Qwen17 raw45 and frames; original public static and all other frozen inputs reused'),row_manifest_SHA=sha(row_manifest),cached_input_qualification=a.cached_check)
    (root/('cached_parent_complete.json'if a.cached_check else 'parent_complete.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='verified_assets'}),flush=True)

if __name__=='__main__':main()
