"""One new 13-parameter joint adapter; default audit, never submits or refits old heads.

The own predictor is the frozen equal mean of MF/Phi/MOSS/Omni candidates.
Original PF modules, readouts, normalizers and original LMY ratio stay frozen.
Training requires a separately frozen source/config/protocol/data registration.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import getpass
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch

V = Path(r'E:\icassp\zhaojiahao\CLIP2-new\algorithms\03_my_improvements\experiments\v10_20261016')
S = V / 'final_selective_risk_20261027'
STAGE = 'JURY_MEAN4_JOINT_ADAPTER_20261027'
ROUTES = [587835, 587836, 587634, 587632]
PF_SHA = 'a32b857ca9bf1da0bb4dcdcdf215bd4034ed74a7bfa060865b100d35b2209b74'
LMY_SHA = '3e663e599420842d5199a95d1e56ad3c5a1089748d1b18bed4cad4968c3c3edb'
W0 = .5609719773532506
TRAIN_ROWS = 3057
TRAIN_IDS_SHA = '66c556dd14afbf5eac8d4e15d174dbb677c23b1b7c35e4ac1c9a6d96ef8368c1'
TRAIN_REGISTRATION = S / 'joint_adapter_only/TRAIN3057_intersection_pre_fit.json'
TRAIN_REGISTRATION_SHA = '516d447282fdf4ceecd8a3ba25f85e239b9d0c568fb69add4b31d1343743010b'
TRAIN_FOLD_PATH = V / 'final_adaptive_20261027/data/gate_train.npz'
TRAIN_FOLD_SHA = '8f38f535ed828461abb44070f120c8766c9ac486e48f63e616155b4a757e8515'
FIXED = dict(seed=142, epochs=80, batch_size=512, optimizer='AdamW',
             learning_rate=.003, weight_decay=.01, foreach=False, CPU_threads=2,
             std_floor=.01, z_clip=5., OOD_z=6., radius=.05)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), SHA=sha(p))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(r):
    p = Path(r['path']).resolve()
    require(sha(p) == r['SHA'], 'Changed registered bytes: ' + str(p))
    return p


def frozen(path, obj):
    path = Path(path)
    require(path.resolve().is_relative_to(S.resolve()), 'Output outside new stage')
    data = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')
    if path.exists():
        require(path.read_bytes() == data, 'Immutable output already differs')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    return ref(path)


def nz(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k].copy() for k in z.files}
    require(all(v.dtype.kind != 'O' for v in out.values()), 'Object data forbidden')
    return out


def csvpred(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    require(len(rows) == 2117 and all(len(r) == 2 for r in rows), 'Expected 2117 native score CSV rows')
    ids = np.asarray([r[0] for r in rows])
    p = np.asarray([float(r[1]) / 100 for r in rows], dtype=np.float64)
    require(len(set(ids)) == len(ids) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid CSV')
    return ids, p


def align(z, ids, id_key='signal_ids'):
    names = list(map(str, z[id_key]))
    require(len(names) == len(set(names)), 'Duplicate source TRAIN IDs')
    ix = {s: i for i, s in enumerate(names)}
    require(all(str(s) in ix for s in ids), 'Missing source IDs')
    return np.asarray([ix[str(s)] for s in ids], dtype=np.int64)


def metadata(z):
    sev = np.asarray(z['severity'])
    count = np.asarray(z['counts'])
    audio = np.asarray(z['audio8'], dtype=np.float64)
    require(audio.shape == (len(sev), 8) and np.isfinite(audio).all(), 'Invalid existing PCA8')
    require(np.isin(sev, [0, 1, 2]).all() and (count > 0).all(), 'Invalid severity or word count')
    return np.column_stack([audio, np.eye(3)[sev.astype(int)], np.log1p(count.astype(float))])


def normalize(x, use):
    return dict(mean=x[use].mean(0).tolist(), std=np.maximum(x[use].std(0), .01).tolist())


def transform(x, norm):
    raw = (x - np.asarray(norm['mean'])) / np.asarray(norm['std'])
    return np.clip(raw, -5, 5), (np.abs(raw) > 6).any(1)


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, ood):
        return torch.where(ood, torch.zeros(x.shape[0], dtype=x.dtype),
                           .05 * torch.tanh(self.linear(x).squeeze(-1)))


def predict(net, z, norm):
    x, ood = transform(metadata(z), norm)
    with torch.inference_mode():
        delta = net(torch.as_tensor(x), torch.as_tensor(ood)).numpy().copy()
    # A saved, rounded canonical VALID J0 is the exact zero point.
    p = z['J0'] + delta * (z['own'] - z['LMY'])
    require(np.isfinite(p).all() and np.isfinite(delta).all(), 'Nonfinite adapter prediction')
    require(np.max(np.abs(delta), initial=0) <= .0500000000001, 'Exceeded absolute 5pp radius')
    require(((W0 + delta >= 0) & (W0 + delta <= 1)).all(), 'Joint weight outside simplex')
    require(((p >= -1e-12) & (p <= 1 + 1e-12)).all(), 'Native prediction outside bounds')
    return p, delta


def validate_metadata_reference(z, r):
    source = nz(checked(r))
    ix = align(source, z['signal_ids'])
    for key in ['audio8', 'severity', 'counts']:
        require(np.array_equal(z[key], source[key][ix]), 'Metadata field identity changed: ' + key)


def prepare(args):
    if not args.config:
        return dict(ready=False, reason='Waiting exact registered joint-only config and data; no fit')
    cp = Path(args.config).resolve()
    require(cp.is_relative_to(S.resolve()) and args.config_sha and sha(cp) == args.config_sha, 'Explicit exact config SHA required')
    c = read(cp)
    require(c['schema'] == 'CLIP2.joint_adapter.train_config.v1' and c['stage_ID'] == STAGE, 'Wrong config scope')
    require(c['source'] == ref(__file__), 'Source identity changed')
    proto, auth = read(checked(c['protocol'])), read(checked(c['authorization']))
    require(proto['schema'] == 'CLIP2.joint_adapter.protocol.v1' and proto['stage_ID'] == STAGE, 'Wrong protocol')
    require(proto['expected_training_rows'] == TRAIN_ROWS, 'New adapter cohort must be the registered 3057 intersection')
    require(sha(TRAIN_REGISTRATION) == TRAIN_REGISTRATION_SHA, 'TRAIN intersection registration changed')
    require(proto['TRAIN_intersection_registration'] == ref(TRAIN_REGISTRATION), 'Wrong fixed intersection registration')
    cohort = read(TRAIN_REGISTRATION)
    require(cohort['expected_training_rows'] == TRAIN_ROWS and len(cohort['excluded_from_new_adapter_fit_only']) == 15, 'Invalid intersection definition')
    require(cohort['original15_data_preserved'] is True and cohort['all_original_models_and_modules_preserved'] is True, 'Original rows or modules must not be deleted')
    require(cohort['selection_used_only_input_ID_availability'] is True and cohort['target_arrays_read_by_this_registration'] is False, 'Intersection must not use labels or loss')
    require(proto['training'] == FIXED and proto['trainable_parameters'] == 13, 'Fixed joint-only formula changed')
    require(proto['own_recipe'] == 'OWN_JURY_MEAN4' and proto['four_route_SIDs'] == ROUTES, 'Own jury recipe changed')
    require(proto['own_weight'] == W0 and proto['only_joint_adapter'] is True, 'Original ratio or scope changed')
    require(proto['original_heads_frozen'] is True and proto['original_internal_coefficients_frozen'] is True, 'Original models must be frozen')
    require(proto['no_POST'] is True and proto['strict_upstream_OOF'] is False, 'Incorrect execution or risk claim')
    require(auth['only_joint_adapter_authorized'] is True and auth['original_internal_changes_authorized'] is False, 'Missing narrow authorization')
    mf = checked(c['data_manifest'])
    m = read(mf)
    require(m['schema'] == 'CLIP2.joint_adapter.data_manifest.v1', 'Wrong data manifest')
    if m['ready'] is not True:
        return dict(ready=False, reason=m.get('blocking_reason', 'TRAIN data identity incomplete'), config=ref(cp), data_manifest=ref(mf))
    data_review = read(checked(c['data_review']))
    require(data_review['accepted'] is True and data_review['data_manifest'] == c['data_manifest'], 'Exact data identity review required')
    require(data_review['LMY_latest_model_identity_matched'] is True and data_review['original_readouts_b_norm_unchanged'] is True, 'Unqualified original data source')
    require(m['LMY_latest_model_identity_matched'] is True, 'Latest LMY TRAIN identity is not matched')
    require(m['TRAIN_prediction_provenance'] == proto['TRAIN_prediction_provenance'], 'TRAIN provenance not exactly registered')
    require(m['whole_system_OOF'] is False, 'Do not fake whole-system OOF')
    tr, va = (nz(checked(m['files'][k])) for k in ['train', 'valid'])
    for z, n in [(tr, TRAIN_ROWS), (va, 2117)]:
        require(len(z['signal_ids']) == len(set(z['signal_ids'])) == n, 'Wrong joint population')
        for name in ['own', 'LMY']:
            z[name] = np.asarray(z[name], dtype=np.float64).reshape(-1)
            require(z[name].shape == (n,) and np.isfinite(z[name]).all() and ((z[name] >= 0) & (z[name] <= 1)).all(), 'Invalid native predictor')
        metadata(z)
    actual_ids = list(map(str, tr['signal_ids']))
    actual_ids_sha = hashlib.sha256(json.dumps(actual_ids, separators=(',', ':')).encode('utf8')).hexdigest()
    require(actual_ids_sha == TRAIN_IDS_SHA == cohort['ordered_TRAIN_ID_SHA'], 'Registered TRAIN ID order changed')
    require(actual_ids == cohort['ordered_TRAIN_signal_ids'], 'Not the exact pre-fit TRAIN intersection')
    require('target' not in va, 'VALID targets are forbidden')
    target = nz(checked(m['TRAIN_target_reference']))
    ix = align(target, tr['signal_ids'])
    require(np.array_equal(tr['target'], target['target'][ix]), 'TRAIN target source changed')
    require(np.isfinite(tr['target']).all() and ((tr['target'] >= 0) & (tr['target'] <= 1)).all(), 'Invalid target unit')
    fold_ref = proto['TRAIN_fold_reference']
    require(fold_ref == m['TRAIN_fold_reference'], 'Protocol and data fold references differ')
    require(fold_ref['path'] == str(TRAIN_FOLD_PATH) and fold_ref['SHA'] == TRAIN_FOLD_SHA,
            'Only the pre-fit fixed artist proxy fold reference is authorized')
    require(fold_ref['ID_key'] == 'signal_ids' and fold_ref['fold_key'] == 'fold', 'Wrong fixed fold fields')
    fs = nz(checked(fold_ref)); fi = align(fs, tr['signal_ids'], fold_ref['ID_key'])
    require(np.array_equal(tr['fold'], fs[fold_ref['fold_key']][fi]) and set(tr['fold']) == set(range(5)),
            'New adapter fixed artist proxy folds changed')
    for split, z in [('TRAIN', tr), ('VALID', va)]:
        validate_metadata_reference(z, m[split + '_metadata_reference'])
    otr = nz(checked(m['own_TRAIN_reference']))
    ix = align(otr, tr['signal_ids'])
    candidates = np.asarray(otr['route_prediction'])[ix]
    pf = np.asarray(otr['PF'])[ix]
    require(candidates.shape == (TRAIN_ROWS, 4) and np.isfinite(candidates).all(), 'Four original TRAIN route predictions missing')
    require(np.max(np.abs(candidates.mean(1) - tr['own'])) < 1e-12, 'TRAIN own is not fixed mean4')
    require(np.max(np.abs(pf + (candidates-pf[:, None]).mean(1) - tr['own'])) < 1e-12, 'PF counted more than once')
    lm = m['LMY_TRAIN_reference']; lz = nz(checked(lm))
    li = align(lz, tr['signal_ids'], lm['ID_key'])
    require(lm['native_scale'] in [1., .01], 'Unregistered LMY score scale')
    lp = np.asarray(lz[lm['prediction_key']], dtype=np.float64).reshape(-1)[li] * lm['native_scale']
    require(np.array_equal(lp, tr['LMY']), 'LMY TRAIN source identity changed')
    own_ids, own = csvpred(checked(m['own_VALID_reference']))
    require(np.array_equal(own_ids, va['signal_ids']) and np.array_equal(own, va['own']), 'Frozen own VALID identity mismatch')
    pfr = m['PF_VALID_reference']; require(pfr['SHA'] == PF_SHA, 'Original PF changed'); pi, _ = csvpred(checked(pfr))
    require(np.array_equal(pi, own_ids), 'Original PF IDs changed')
    routes = m['four_route_VALID_references']
    require([r['submission_id'] for r in routes] == ROUTES, 'Route membership/order changed')
    values = []
    for route in routes:
        ri, rp = csvpred(checked(route)); require(np.array_equal(ri, own_ids), 'Route IDs changed'); values.append(rp)
    require(np.max(np.abs(np.mean(values, axis=0)-own)) <= 5.1e-13, 'Frozen VALID own is not mean4 within declared decimal rounding')
    lmr = m['LMY_VALID_reference']; require(lmr['SHA'] == LMY_SHA, 'Latest LMY export changed')
    lids, lv = csvpred(checked(lmr)); lj = align(dict(signal_ids=lids), own_ids)
    require(np.array_equal(lv[lj], va['LMY']), 'LMY VALID order/values changed')
    ji, j0 = csvpred(checked(m['joint_zero_VALID_reference']))
    require(np.array_equal(ji, own_ids), 'New own joint zero IDs changed')
    require(np.max(np.abs(j0-(W0*own+(1-W0)*va['LMY']))) <= 5.1e-13, 'New mean4 zero-joint does not use original ratio')
    va['J0'] = j0
    tr['J0'] = W0*tr['own'] + (1-W0)*tr['LMY']
    require(m['original_readouts_b_norm_unchanged'] is True, 'Old readout coefficients or normalizers changed')
    review = read(checked(c['technical_review']))
    require(review['accepted'] is True and review['source'] == c['source'], 'Exact source technical review required')
    root = Path(c['output_root']).resolve()
    require(root.is_relative_to(S.resolve()), 'Run output outside authorized new stage')
    binding = {key: c[key] for key in ['source', 'protocol', 'authorization', 'data_manifest', 'data_review', 'technical_review']}
    binding['config'] = ref(cp)
    binding['TRAIN_intersection_registration'] = ref(TRAIN_REGISTRATION)
    return dict(ready=True, config=c, protocol=proto, authorization=auth, data=m,
                train=tr, valid=va, output=root, binding=binding,
                training_authorized=proto['training_authorized'] is True and auth['training_authorized'] is True)


def fit_once(q, fold):
    root = q['output'] / 'fits' / ('full' if fold < 0 else 'fold' + str(fold))
    result_path = root / 'result.json'
    tr, va = q['train'], q['valid']
    if result_path.exists():
        r = read(result_path)
        require(r['binding'] == q['binding'] and r['fold'] == fold, 'Existing fit binding changed')
        checked(r['model']); checked(r['predictions'])
        require(r['checkpoint_reload_exact'] is True, 'Existing fit has no replay proof')
        return r
    root.mkdir(parents=True, exist_ok=True)
    intent_path = root / 'intent.json'
    require(not intent_path.exists(), 'Unknown prior fit intent is reserved; never automatically refit')
    use = np.ones(len(tr['signal_ids']), dtype=bool) if fold < 0 else tr['fold'] != fold
    held = ~use
    frozen(intent_path, dict(UTC=utc(), PID=os.getpid(), user=getpass.getuser(), argv=sys.argv,
                            fold=fold, binding=q['binding'], fit_IDs=list(map(str, tr['signal_ids'][use])),
                            held_IDs=list(map(str, tr['signal_ids'][held])), one_fit_only=True))
    norm = normalize(metadata(tr), use)
    x, ood = transform(metadata(tr), norm)
    torch.manual_seed(142); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    net = Adapter()
    require(sum(p.numel() for p in net.parameters()) == 13, 'Wrong parameter count')
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.01, foreach=False)
    tx = torch.as_tensor(x[use]); to = torch.as_tensor(ood[use])
    base = torch.as_tensor(tr['J0'][use]); diff = torch.as_tensor((tr['own']-tr['LMY'])[use])
    target = torch.as_tensor(np.asarray(tr['target'][use], dtype=np.float64))
    rng = np.random.default_rng(142); updates = 0; start = time.monotonic()
    for epoch in range(80):
        order = rng.permutation(len(tx))
        for batch in np.array_split(order, int(np.ceil(len(tx)/512))):
            delta = net(tx[batch], to[batch])
            p = base[batch] + delta*diff[batch]
            loss = (p-target[batch]).square().mean()
            require(bool(torch.isfinite(loss)), 'Nonfinite TRAIN loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); updates += 1
    net.eval()
    vh = {k: v[held] for k, v in tr.items() if isinstance(v, np.ndarray) and len(v) == len(use)}
    hp, hd = predict(net, vh, norm); vp, vd = predict(net, va, norm)
    model_path = root/'model.pt'
    torch.save(dict(state=net.state_dict(), normalization=norm, parameters=13,
                    binding=q['binding'], fold=fold, training=FIXED), model_path)
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    reload = Adapter(); reload.load_state_dict(saved['state'], strict=True); reload.eval()
    rh, dh = predict(reload, vh, saved['normalization']); rv, dv = predict(reload, va, saved['normalization'])
    exact = all(np.array_equal(a, b) for a, b in [(hp, rh), (hd, dh), (vp, rv), (vd, dv)])
    require(exact, 'Checkpoint reload predictions not identical')
    pred = root/'predictions.npz'
    np.savez_compressed(pred, held_signal_ids=tr['signal_ids'][held], held_prediction=hp,
                        held_delta=hd, VALID_signal_ids=va['signal_ids'], VALID_prediction=vp, VALID_delta=vd)
    r = dict(UTC=utc(), binding=q['binding'], fold=fold, fit_rows=int(use.sum()),
             held_rows=int(held.sum()), parameters=13, model=ref(model_path), predictions=ref(pred),
             normalization=norm, epochs=80, optimizer_updates=updates, checkpoint_reload_exact=exact,
             seconds=time.monotonic()-start, TRAIN_prediction_provenance=q['data']['TRAIN_prediction_provenance'],
             whole_system_OOF=False, original_head_fits=0, VALID_labels_read=False, POSTs=0)
    frozen(result_path, r)
    return r


def train(q):
    require(q.get('ready') is True and q['training_authorized'] is True, 'Data or exact training authorization incomplete')
    root = q['output']; terminal_path = root/'terminal.json'
    if terminal_path.exists():
        t = read(terminal_path); require(t['binding'] == q['binding'] and t['complete'] is True, 'Terminal binding changed')
        checked(t['candidate']); checked(t['report']); return dict(existing_terminal_reused=ref(terminal_path), new_fits=0)
    root.mkdir(parents=True, exist_ok=True)
    intent = root/'run_intent.json'
    require(not intent.exists(), 'Existing run intent is reserved; do not start a second trainer or retry unknown fits')
    frozen(intent, dict(UTC=utc(), PID=os.getpid(), user=getpass.getuser(), binding=q['binding'], argv=sys.argv))
    records = []
    for fold in [0, 1, 2, 3, 4, -1]:
        for r in q['binding'].values(): checked(r)
        record = fit_once(q, fold); records.append(record)
        print(json.dumps(dict(completed_fit=fold, seconds=record['seconds'], original_heads_trained=0)), flush=True)
    full = nz(checked(records[-1]['predictions']))
    candidate = root/'candidate/valid_predictions.csv'; candidate.parent.mkdir(exist_ok=True)
    if np.count_nonzero(full['VALID_delta']) == 0:
        payload = checked(q['data']['joint_zero_VALID_reference']).read_bytes()
    else:
        stream = io.StringIO(newline='')
        csv.writer(stream, lineterminator='\r\n').writerows(zip(q['valid']['signal_ids'],
                  [format(float(v)*100, '.10f') for v in full['VALID_prediction']]))
        payload = stream.getvalue().encode('utf8')
    with candidate.open('xb') as f: f.write(payload)
    held = np.empty(TRAIN_ROWS); delta = np.empty(TRAIN_ROWS)
    for fold, r in enumerate(records[:5]):
        zz = nz(checked(r['predictions'])); ix = align(q['train'], zz['held_signal_ids'])
        require(np.array_equal(q['train']['fold'][ix], np.full(len(ix), fold)), 'Held fold assignment changed')
        held[ix] = zz['held_prediction']; delta[ix] = zz['held_delta']
    hp = root/'new_adapter_held_diagnostic.npz'
    np.savez_compressed(hp, signal_ids=q['train']['signal_ids'], prediction=held, delta=delta,
                        target=q['train']['target'], fold=q['train']['fold'])
    report_path = root/'candidate/report.json'
    report = dict(UTC=utc(), stage_ID=STAGE, binding=q['binding'], candidate=ref(candidate),
                  own_recipe='OWN_JURY_MEAN4', own_VALID_reference=q['data']['own_VALID_reference'],
                  frozen_original_own_weight=W0, frozen_LMY_weight=1-W0, absolute_radius=.05,
                  trainable_parameters=13, new_adapter_fit_count=6, original_head_fits=0,
                  TRAIN_rows=TRAIN_ROWS, VALID_rows=2117, held_diagnostic=ref(hp),
                  original_own_TRAIN_rows=3072, excluded_from_new_adapter_fit_only=15,
                  original_rows_and_all_modules_preserved=True,
                  TRAIN_intersection_registration=ref(TRAIN_REGISTRATION),
                  TRAIN_fold_reference=q['protocol']['TRAIN_fold_reference'],
                  TRAIN_fold_scope='Fixed artist proxy fold vector; conditional adapter-held diagnostic, not old lyric CV or whole-system OOF',
                  conditional_adapter_held_MSE_score=float(np.mean((held-q['train']['target'])**2)*10000),
                  TRAIN_prediction_provenance=q['data']['TRAIN_prediction_provenance'],
                  no_hyperparameter_search=True, no_original_internal_changes=True, no_original_b_norm_refit=True,
                  whole_system_OOF=False, zero_overfit_certified=False, official_submission_eligible=False,
                  limitation='Adapter-held only with frozen conditional upstream. Baseline VALID-history is retained by user direction; no latest-peer OOF or low-risk certification is invented.',
                  actual_official_result=None, POSTs=0, EVAL=False)
    frozen(report_path, report)
    terminal = dict(UTC=utc(), complete=True, binding=q['binding'], candidate=ref(candidate),
                    report=ref(report_path), fit_results=[ref(q['output']/'fits'/('full' if k<0 else 'fold'+str(k))/'result.json') for k in [0,1,2,3,4,-1]],
                    new_fits=6, original_head_fits=0, no_POST=True, final_adoption=False)
    return dict(terminal=frozen(terminal_path, terminal), new_fits=6, POSTs=0)


def self_test():
    checks = []
    torch.set_num_threads(2)
    z = dict(signal_ids=np.asarray(['a','b','c']), audio8=np.zeros((3,8)),
             severity=np.asarray([0,1,2]), counts=np.asarray([2,3,4]),
             own=np.asarray([.2,.8,.4]), LMY=np.asarray([.8,.3,.6]))
    z['J0'] = W0*z['own']+(1-W0)*z['LMY']
    x = metadata(z); norm = normalize(x, np.ones(3,dtype=bool)); net = Adapter()
    p, d = predict(net,z,norm); checks.append(('zero_exact',np.array_equal(p,z['J0']) and np.count_nonzero(d)==0))
    with torch.no_grad(): net.linear.bias.fill_(100)
    p,d=predict(net,z,norm); checks.append(('weight_box_and_sum',np.max(np.abs(d))<=.05 and np.allclose((W0+d)+(1-W0-d),1)))
    zz={k:v.copy() for k,v in z.items()}; zz['audio8'][0,0]=100
    p,d=predict(net,zz,norm); checks.append(('OOD_exact_original',p[0]==zz['J0'][0] and d[0]==0))
    net=Adapter(); tx,oo=transform(x,norm); y=net(torch.as_tensor(tx),torch.as_tensor(oo)); (y*torch.tensor([1.,2.,3.])).sum().backward()
    checks.append(('nonzero_gradient_at_zero',net.linear.bias.grad.abs().item()>0))
    altered=x.copy();altered[2]=1e6; use=np.asarray([True,True,False]);checks.append(('fit_only_norm',normalize(x,use)==normalize(altered,use)))
    buffer=io.BytesIO();torch.save(net.state_dict(),buffer);buffer.seek(0);r=Adapter();r.load_state_dict(torch.load(buffer,weights_only=True));a=predict(net,z,norm);b=predict(r,z,norm)
    checks.append(('checkpoint_memory_exact',all(np.array_equal(i,j)for i,j in zip(a,b))))
    pf=np.asarray([.2,.5]);routes=np.asarray([[.21,.23,.18,.22],[.51,.48,.52,.49]])
    checks.append(('PF_once_mean4',np.allclose(pf+(routes-pf[:,None]).mean(1),routes.mean(1))))
    require(all(ok for _,ok in checks),'Self-test failed')
    return dict(passed=True,checks=[dict(name=n,passed=bool(ok))for n,ok in checks],optimizer_updates=0,POSTs=0,original_model_forwards=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config');ap.add_argument('--config-sha')
    ap.add_argument('--mode',choices=['audit','self-test','train'],default='audit')
    args=ap.parse_args()
    if args.mode=='self-test':result=self_test()
    else:
        q=prepare(args)
        if args.mode=='audit':result=dict(audit_passed=True,ready=q['ready'],reason=q.get('reason'),training_authorized=q.get('training_authorized',False),new_fits=0,POSTs=0)
        else:result=train(q)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
