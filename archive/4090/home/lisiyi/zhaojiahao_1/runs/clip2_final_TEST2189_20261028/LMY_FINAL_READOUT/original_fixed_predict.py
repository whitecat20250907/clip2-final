"""Predict-only reconstruction of the exact latest LMY frozen export ancestry.

Default audit reads the registration only. Execute is separately authorized.
No peer imports, training, labels, CV, optimizer, API or GPU use. All writes are
under the registration's owned output directory; peer files are read-only.
"""
import os, sys, pathlib, json, hashlib, argparse, csv, io, datetime
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--manifest-sha', required=True)
    ap.add_argument('--mode', choices=['audit', 'execute'], default='audit')
    a = ap.parse_args()
    mp = pathlib.Path(a.manifest).resolve()
    assert sha(mp) == a.manifest_sha
    m = json.loads(mp.read_text(encoding='utf-8'))
    assert m['schema'] == 'CLIP2.latest_LMY.fixed_TRAIN_prediction.v1'
    assert m['source_SHA'] == sha(__file__)
    assert m['expected_VALID_CSV_SHA'] == '3e663e599420842d5199a95d1e56ad3c5a1089748d1b18bed4cad4968c3c3edb'
    assert m['training_updates'] == m['old_CV_recomputations'] == 0
    assert m['maximum_replay_gap_native'] == 1e-6
    assert m['maximum_replay_gap_0to100_points'] == 1e-4
    out = pathlib.Path(m['output_root']).resolve()
    owner = pathlib.Path(m['owned_root']).resolve()
    assert os.path.commonpath([str(out), str(owner)]) == str(owner)
    assert out != owner
    assert m['TRAIN_parent_provenance'] == 'original frozen FIT/held public feature caches; fixed full-fit deployment heads; conditional/in-sample, not whole-pipeline OOF'
    assert len(m['TRAIN_signal_ids']) == len(set(m['TRAIN_signal_ids'])) == 3057
    assert len(m['omitted_original_TRAIN_IDs_due_only_to_input_availability']) == 15
    assert not set(m['TRAIN_signal_ids']).intersection(m['omitted_original_TRAIN_IDs_due_only_to_input_availability'])
    assert set(m['TRAIN_signal_ids']) | set(m['omitted_original_TRAIN_IDs_due_only_to_input_availability']) == set(m['original_3072_TRAIN_signal_ids'])
    assert len(m['VALID_signal_ids']) == len(set(m['VALID_signal_ids'])) == 2117
    assert not set(m['TRAIN_signal_ids']).intersection(m['VALID_signal_ids'])
    assert all(r['SHA'] and len(r['SHA']) == 64 for r in m['assets'].values())
    print(json.dumps(dict(audit_passed=True, mode=a.mode, assets=len(m['assets']), TRAIN=3057, VALID=2117, fit=0, CV=0, GPU=0)))
    if a.mode != 'execute':
        return
    assert m['execute_authorized'] is True
    assert m['input_qualification_ready'] is True
    assert owner.stat().st_uid == os.getuid() == m['expected_UID']
    # No import of peer common_v1: it creates peer runtime directories.
    out.mkdir(parents=True, exist_ok=True)
    for key in ['HF_HOME', 'HF_HUB_CACHE', 'XDG_CACHE_HOME', 'TMPDIR', 'TORCH_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'PYTHONPYCACHEPREFIX']:
        p = out / 'runtime' / key
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from torch import nn
    from scipy.special import expit, logit
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verified = {}
    cache = {}

    def pin(key):
        r = m['assets'][key]
        p = pathlib.Path(r['path'])
        assert p.is_file() and sha(p) == r['SHA'], key
        verified[key] = dict(path=str(p), SHA=r['SHA'])
        return p

    def load(key):
        if key not in cache:
            # NpzFile is lazy: do not read unused labels/targets from a source
            # that also contains public input arrays. Only named fields below
            # are ever materialized.
            cache[key] = np.load(pin(key), allow_pickle=False)
        return cache[key]

    def take(key, ids, names, id_key='signals'):
        z = load(key)
        ss = z[id_key].tolist()
        assert len(ss) == len(set(ss)), key
        ix = {s: i for i, s in enumerate(ss)}
        assert set(ids) <= set(ix), (key, 'missing IDs', sorted(set(ids) - set(ix))[:5])
        ii = np.asarray([ix[s] for s in ids], dtype=np.int64)
        return {k: z[k][ii].copy() for k in names}

    def q39_train(seed, ids):
        z = load('q39_train' + str(seed))
        ss = z['FIT_signals'].tolist() + z['held_signals'].tolist()
        assert len(ss) == len(set(ss)) == 16180
        xx = np.concatenate([z['FIT_features'], z['held_features']], 0)
        ix = {s: i for i, s in enumerate(ss)}
        assert set(ids) <= set(ix)
        return xx[[ix[s] for s in ids]].copy()

    def standard(x, nm):
        return np.clip((np.where(np.isfinite(x), x, nm['median']) - nm['mean']) / nm['std'], -6, 6)

    def rr(x, key, sigmoid=False):
        st = load(key)
        z = standard(x, st)
        v = np.c_[np.ones(len(x)), z] @ st['coef']
        return (expit(v) if sigmoid else np.clip(v, 0, 1)), z

    def wordnet(x, key, anchor=None, legal=None, clamp=1e-5, branch=False):
        if key.startswith('WF_head'):
            # Original WF checkpoint also stores inert NumPy normalization
            # metadata. Its exact pinned bytes were inspected before launch;
            # allow only the NumPy array/dtype constructors, never arbitrary
            # checkpoint globals or weights_only=False.
            def numpy_reconstruct_alias(*args):
                return np.core.multiarray._reconstruct(*args)
            numpy_reconstruct_alias.__module__='numpy._core.multiarray'
            numpy_reconstruct_alias.__name__='_reconstruct'
            safe=[np.core.multiarray._reconstruct,numpy_reconstruct_alias,np.ndarray,np.dtype,
                  type(np.dtype('float64')),type(np.dtype('float32')),
                  type(np.dtype('int64')),type(np.dtype('int32'))]
            with torch.serialization.safe_globals(safe):
                cp=torch.load(pin(key),map_location='cpu',weights_only=True)
        else:
            cp = torch.load(pin(key), map_location='cpu', weights_only=True)
        w = cp['weights']
        if branch:
            w = {k[len('branch.'):]: v for k, v in w.items() if k.startswith('branch.')}
        net = nn.Sequential(nn.Linear(w['0.weight'].shape[1], w['0.weight'].shape[0]), nn.GELU(), nn.Linear(w['0.weight'].shape[0], 1))
        net.load_state_dict(w)
        net.eval().requires_grad_(False)
        with torch.inference_mode():
            p = net(torch.tensor(x, dtype=torch.float32)).squeeze(-1)
            if anchor is not None:
                base = torch.logit(torch.tensor(anchor, dtype=torch.float32).clamp(clamp, 1-clamp))
                p = p + (base[:, None] if p.ndim == 2 and base.ndim == 1 else base)
            p = p.sigmoid()
            if legal is not None:
                lt = torch.tensor(legal, dtype=torch.float32)
                p = (p * lt).sum(1) / lt.sum(1)
        return p.double().numpy()

    def pool(x, legal):
        assert legal.shape == x.shape[:2] and (legal.sum(1) > 0).all()
        mu = (x * legal[..., None]).sum(1) / legal.sum(1)[:, None]
        sd = np.sqrt(np.maximum(((x-mu[:, None, :])**2 * legal[..., None]).sum(1) / legal.sum(1)[:, None], 0))
        return np.c_[mu, sd]

    def frozen_forward(ids, role):
        is_train = role == 'TRAIN'
        if is_train:
            loc = take('local81', ids, ['local', 'legal']); local = loc['local']; legal = loc['legal']
            native = take('native48', ids, ['features'])['features']
            old32 = take('raw_old32', ids, ['features32'])['features32']
            new32 = take('raw_new32', ids, ['features32'])['features32']
            h50 = take('human50', ids, ['frozen', 'human'])
        else:
            z = take('valid_native81', ids, ['local81', 'legal', 'native48']); local=z['local81']; legal=z['legal']; native=z['native48']
            zz = take('valid_tdt', ids, ['old32','new32','frozen50','human50','legal'])
            assert np.array_equal(zz['legal'], legal)
            old32=zz['old32'];new32=zz['new32'];h50={'frozen':zz['frozen50'],'human':zz['human50']}
        assert local.shape == (len(ids),12,81) and legal.shape == (len(ids),12)
        assert np.array_equal(legal, np.arange(12)[None,:] < legal.sum(1)[:,None])
        extra = np.concatenate([old32,new32], -1); pooled = pool(extra,legal)
        mc=take('main_cache',ids,['x45','words','parent44','parent45']);assert np.array_equal(mc['words'],legal)
        f96=take('main_confidence96',ids,['normalized96'])['normalized96'];main115=np.concatenate([mc['x45'],f96[...,:70]],-1)
        wn=load('norm_word');hn=load('norm_homo');an=load('norm_AF');sn=load('norm_SR');wfn=load('norm_WF')
        zl30=np.clip((local[...,:30]-wn['word_mean'])/wn['word_std'],-6,6)
        zl81=np.clip((local-hn['word_mean'])/hn['word_std'],-6,6)
        ze=np.clip((old32-an['mean'])/an['std'],-6,6).astype(np.float32)
        sl=np.clip((local-sn['local_mean'])/sn['local_std'],-6,6).astype(np.float32)
        se=np.clip((old32-sn['extra_mean'])/sn['extra_std'],-6,6).astype(np.float32)
        we=np.clip((extra-wfn['mean'])/wfn['std'],-6,6).astype(np.float32)
        outs={};gfs={};stages={}
        for seed in [44,45]:
            s=str(seed)
            q39=q39_train(seed,ids) if is_train else take('q39_valid'+s,ids,['features'])['features']
            stx=take('stereo_x'+s if is_train else 'stereo_valid'+s,ids,['features'])['features']
            core=take('main_sealed'+s,ids,['prediction'])['prediction'].astype(np.float64)
            components=[('qwen39',q39)]
            for fam in ['PARAKEET','GRANITE']:
                k=('basis_x_' if is_train else 'basis_valid_')+fam+s
                components.append((fam.lower()+'35',take(k,ids,['features'])['features']))
            ep=np.column_stack([rr(xx,'basis_head_'+name+s)[0] for name,xx in components]);uniform=ep.mean(1)
            stereo=rr(stx,'stereo_head'+s)[0];pStereo=.9*core+.1*stereo;pBasis=.9*pStereo+.1*uniform
            z109=standard(np.concatenate([xx for _,xx in components],1),load('gate_norm'+s))
            calanchor=torch.tensor(ep,dtype=torch.float32).mean(1).double().numpy()
            qGate=wordnet(z109,'gate_head'+s,calanchor)
            pGate=np.clip(pBasis+.1*(qGate-uniform),0,1)
            base=np.c_[q39,native];q87,zg=rr(base,'base87_head'+s)
            x117=np.concatenate([np.repeat(zg[:,None,:],12,1),zl30],2);x168=np.concatenate([np.repeat(zg[:,None,:],12,1),zl81],2)
            qWord=wordnet(x117,'word_head'+s,q87,legal);qHomo=wordnet(x168,'homo_head'+s,q87,legal)
            qDirect=wordnet(x168,'direct_head'+s,None,legal);qContinue=wordnet(x168,'continue_head'+s,None,legal)
            pDelta=np.clip(pGate+.2*(qWord-ep[:,0]),0,1);pHomo=np.clip(pDelta+.2*(qHomo-qWord),0,1)
            pDirect=np.clip(pHomo+.1*(.5*qDirect+.5*qHomo-qHomo),0,1);pBest=np.clip(pDirect+.05*(qContinue-qDirect),0,1)
            qnew=rr(np.c_[base,h50['human']],'HS_new'+s)[0];qc=rr(np.c_[base,h50['frozen']],'HS_control'+s)[0]
            p=np.clip(pBest+.1*((.5*qnew+.5*qc)-qc),0,1)
            zg32=np.repeat(zg.astype(np.float32)[:,None,:],12,1);zl32=zl81.astype(np.float32)
            q168=wordnet(np.concatenate([zg32,zl32],-1),'homo_head'+s,q87,legal)
            q200=wordnet(np.concatenate([zg32,zl32,ze],-1),'AF_head'+s,q87,legal)
            p=np.clip(p+.1*((.5*q200+.5*q168)-q168),0,1)
            old=take('main_sealed'+s,ids,['word_probability','prediction','words']);assert np.array_equal(old['words'],legal)
            pw=old['word_probability'];qold=old['prediction'].astype(np.float64)
            bl=torch.logit(torch.tensor(pw,dtype=torch.float32).clamp(1e-5,1-1e-5)).numpy()
            qsr=wordnet(np.concatenate([sl,se,bl[...,None]/6],-1),'SR_head'+s,pw,legal)
            p=np.clip(p+.25*((.5*qsr+.5*qold)-qold),0,1)
            x215=np.c_[base,pooled];q215=rr(x215,'GM_head'+s)[0]
            p=np.clip(p+.05*(q215-q87),0,1);p=np.clip(.95*p+.05*q215,0,1)
            qwarm=wordnet(np.concatenate([main115,we],-1),'WF_head'+s,mc['parent'+s],legal,clamp=1e-7,branch=True)
            p=np.clip(p+.2*(qwarm-qold),0,1)
            qinv=rr(x215,'RF_head'+s)[0];p=np.clip(p+.2*((.5*qinv+.5*q215)-q215),0,1)
            qglm=rr(x215,'GF_head'+s,True)[0];p=np.clip(p+.05*((.5*qglm+.5*q215)-q215),0,1)
            gfs[seed]=p
            if is_train:
                ctx=take('x920',ids,['x920_seed'+s])['x920_seed'+s]
                z0=take('old_mvdr',ids,['beam18','legal']);z1=take('new_mvdr',ids,['features','legal'])
                assert np.array_equal(z0['legal'],legal) and np.array_equal(z1['legal'],legal)
                x=np.c_[ctx,pool(z0['beam18'],legal),pool(z1['features'],legal)]
            else:
                vb=take('valid_base920',ids,['base'+s,'legal']);vf=take('valid_mvdr',ids,['features','legal'])
                assert np.array_equal(vb['legal'],legal) and np.array_equal(vf['legal'],legal)
                x=np.c_[vb['base'+s],pool(vf['features'][...,:18],legal),pool(vf['features'][...,18:],legal)]
            assert x.shape==(len(ids),1004)
            q=rr(x,'MF_count1004'+s,True)[0]
            outs[seed]=expit(.5*(logit(np.clip(p,1e-6,1-1e-6))+logit(np.clip(q,1e-6,1-1e-6))))
            stages[s]=dict(parent_min=float(p.min()),parent_max=float(p.max()),Count_min=float(q.min()),Count_max=float(q.max()))
            print(json.dumps(dict(role=role,seed=seed,rows=len(ids),complete=True)),flush=True)
        return (outs[44]+outs[45])/2,gfs,stages

    # Recheck all asset bytes and every required ID mapping before claiming
    # the unique prediction intent. No model forward is needed for this.
    for key in m['assets']:
        pin(key)
    for key in ['local81','native48','raw_old32','raw_new32','human50','main_cache','main_confidence96','main_sealed44','main_sealed45','stereo_x44','stereo_x45','basis_x_PARAKEET44','basis_x_PARAKEET45','basis_x_GRANITE44','basis_x_GRANITE45','x920','old_mvdr','new_mvdr']:
        ss=load(key)['signals'].tolist()
        assert len(ss)==len(set(ss)) and set(m['TRAIN_signal_ids'])<=set(ss),key
    for seed in [44,45]:
        z=load('q39_train'+str(seed));ss=z['FIT_signals'].tolist()+z['held_signals'].tolist()
        assert len(ss)==len(set(ss))==16180 and set(m['TRAIN_signal_ids'])<=set(ss)
    for key in ['valid_native81','valid_tdt','q39_valid44','q39_valid45','stereo_valid44','stereo_valid45','valid_GF','valid_base920','valid_mvdr']:
        assert load(key)['signals'].tolist()==m['VALID_signal_ids'],key
    with (out/'predict.intent.json').open('x',encoding='utf-8')as f:
        json.dump(dict(manifest_SHA=sha(mp),source_SHA=sha(__file__),PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_updates=0),f)
    # Reproduce deployed VALID through every original frozen ancestor before
    # accepting TRAIN. No VALID label is in the manifest, loaded, or measured.
    valid_ids=m['VALID_signal_ids'];qv,gfv,vstage=frozen_forward(valid_ids,'VALID')
    fv=take('valid_GF',valid_ids,['GF_seed44','GF_seed45'])
    gf_gap={str(s):float(abs(gfv[s]-fv['GF_seed'+str(s)]).max()) for s in [44,45]}
    assert max(gf_gap.values()) <= m['maximum_replay_gap_native'],gf_gap
    cp=pin('valid_csv');assert sha(cp)==m['expected_VALID_CSV_SHA']
    rows=list(csv.reader(cp.open(newline='')));assert [r[0]for r in rows]==valid_ids
    cached_valid=np.asarray([float(r[1])/100 for r in rows])
    gap=float(abs(qv-cached_valid).max());assert gap <= m['maximum_replay_gap_native'],gap
    b=io.StringIO(newline='');csv.writer(b,lineterminator='\n').writerows(zip(valid_ids,[format(float(v*100),'.12f')for v in qv]))
    exactcsv=hashlib.sha256(b.getvalue().encode()).hexdigest()==m['expected_VALID_CSV_SHA']
    qt,gft,tstage=frozen_forward(m['TRAIN_signal_ids'],'TRAIN')
    assert qt.shape==(3057,) and np.isfinite(qt).all() and qt.min()>=0 and qt.max()<=1
    # Output native TRAIN; VALID retains exact already-official CSV values.
    for fn,ids,q in [('TRAIN_fixed_prediction.npz',m['TRAIN_signal_ids'],qt),('VALID_bound_prediction.npz',valid_ids,cached_valid)]:
        with (out/fn).open('xb')as f:np.savez_compressed(f,signal_ids=np.asarray(ids),prediction=q)
    proof=dict(schema='CLIP2.latest_LMY.fixed_TRAIN_prediction.complete.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_SHA=sha(__file__),manifest_SHA=sha(mp),model_identity_VALID_CSV_SHA=m['expected_VALID_CSV_SHA'],TRAIN_rows=3057,VALID_rows=2117,TRAIN_parent_provenance=m['TRAIN_parent_provenance'],whole_pipeline_OOF=False,training_updates=0,old_CV_recomputations=0,VALID_labels_read=False,GPU_used=False,VALID_GF_replay_maxgaps=gf_gap,VALID_final_replay_maxgap_native=gap,VALID_full_chain_replay_CSV_byte_identical=exactcsv,VALID_output_uses_exact_existing_official_CSV=True,TRAIN_stages=tstage,VALID_stages=vstage,verified_assets=verified,outputs={n:dict(path=str(out/n),SHA=sha(out/n))for n in ['TRAIN_fixed_prediction.npz','VALID_bound_prediction.npz']})
    with (out/'complete.json').open('x',encoding='utf-8')as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=True,VALID_gap=gap,VALID_CSV_byte_identical=exactcsv,TRAIN=3057,fit=0)),flush=True)


if __name__=='__main__':
    main()
