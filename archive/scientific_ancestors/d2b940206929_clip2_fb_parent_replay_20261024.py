"""Frozen PRIOR_FIXED graph replay, identity verification only; no fit or scores.

Outputs the 3072-row conditional TRAIN mapping needed for residual structure
diagnostics. This is not recursively cross-fitted under any new split protocol.
"""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from scipy.special import expit, logit

V = Path(r'E:/icassp/zhaojiahao/CLIP2-new/algorithms/03_my_improvements/experiments/v10_20261016')
OUT = V/'next_fixed_blocks_20261024/train_parent_audit'
REPLAY = V/'fourth_round_20261023/V11_replay'
PF = V/'third_round_20261023/Z13_attribution_20261023/PRIOR_FIXED'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arr(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def anchor(proxy, mask, parent):
    # Preserve exactly the original 70-iteration inference operation.
    l = logit(np.clip(proxy, 1e-8, 1-1e-8))
    lo = np.full(len(l), -50.)
    hi = -lo
    for _ in range(70):
        mid = (lo+hi)/2
        q = (expit(l+mid[:, None])*mask).sum(1)/mask.sum(1)
        lo = np.where(q < parent, mid, lo)
        hi = np.where(q < parent, hi, mid)
    return expit(l+((lo+hi)/2)[:, None])


def meanword(p, mask):
    return (p*mask).sum(1)/mask.sum(1)


def standard(x, m):
    return np.clip((x-np.asarray(m['mean']))/np.asarray(m['std']), -5, 5)


def sentence_apply(base, x, m):
    # Existing Y-node standardization was float32; preserve this detail.
    a = np.clip((x-np.asarray(m['mean'], dtype=x.dtype))/np.asarray(m['std'], dtype=x.dtype), -5, 5)
    return expit(logit(np.clip(base, 1e-5, 1-1e-5))+.2*np.tanh(a@np.asarray(m['weights'])))


def word_apply(base, proxy, mask, x, m):
    p = anchor(proxy, mask, base)
    l = logit(np.clip(p, 1e-9, 1-1e-9))
    v = .2*np.tanh(standard(x, m)@np.asarray(m['weights']))
    if m.get('transform', 'logit') == 'power':
        q = -np.expm1(np.exp(v)*np.log1p(-np.clip(expit(l), 1e-12, 1-1e-12)))
    else:
        q = expit(l+v)
    return base+m.get('scale', 1.)*(meanword(q, mask)-base)


def z2_apply(base, proxy, mask, x, m):
    p = anchor(proxy, mask, base)
    xx = np.concatenate([np.ones((*mask.shape, 1)), standard(x, m)], 2)
    q = -np.expm1(np.exp(.2*np.tanh(xx@np.asarray(m['weights'])))*np.log1p(-np.clip(p, 1e-12, 1-1e-12)))
    return meanword(q, mask)


def pipeline(z, models, coefficients, split, ybase, fixed_prior):
    base = z['train_parent_conditional' if split == 'train' else 'valid_parent_frozen587135']
    proxy, mask = z[split+'_proxy'], z[split+'_mask']
    branch = {k: sentence_apply(ybase, z[split+'_'+k], models[k]) for k in ['Y2', 'Y2-TF', 'Y3', 'Y4']}
    pre = base+sum(coefficients[k]*(branch[k]-base) for k in branch)
    er1 = z['train_ER01' if split == 'train' else 'valid_ER01_frozen_cache']
    er2 = z['train_ER02' if split == 'train' else 'valid_ER02_frozen_cache']
    first = pre+coefficients['ER01']*(er1-base)
    before_z2 = first+(er2-er1)
    z2 = z2_apply(before_z2, proxy, mask, z[split+'_Z2'], models['Z2'])
    third = {k: word_apply(z2, proxy, mask, z[split+'_'+k], models[k]) for k in ['Z8', 'Z11', 'Z9']}
    composed = z2+sum(q-z2 for q in third.values())
    prior_model = dict(models['Z13'], weights=[fixed_prior['fixed_weight']], mean=fixed_prior['mean'], std=fixed_prior['std'], scale=1.)
    result = word_apply(composed, proxy, mask, z[split+'_Z13'], prior_model)
    return result, dict(root=base, first_composition=first, before_Z2=before_z2, Z2=z2, COMPOSE=composed, PRIOR_FIXED=result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    complete = OUT/'parent_replay_complete.json'
    if complete.exists():
        print(complete.read_text(encoding='utf8'))
        return
    inputs = [Path(__file__), REPLAY/'joint_fit_inputs_conditional.npz', REPLAY/'frozen_models.json',
              V/'evidence_relations_20261022/ER01/train.npz', V/'evidence_relations_20261022/ER01/valid.npz',
              PF/'model.json', PF/'candidate/valid_predictions.csv',
              V/'fourth_round_20261023/Q0_Q5/train_metadata_linkage.jsonl',
              V/'fourth_round_20261023/Q0_Q5/folds_song_proxy_all_clean_train.npz',
              V/'v12_music_20261023/P0_parent_OOF_20261024/parent_audit_complete.json',
              V/'third_round_20261023/Z13_attribution_20261023/PRIOR_FIXED/candidate/report.json']
    source_sha = {str(p): sha(p) for p in inputs}
    z = arr(REPLAY/'joint_fit_inputs_conditional.npz')
    frozen = read(REPLAY/'frozen_models.json')
    models, coeff = frozen['models'], frozen['first_composition_coefficients']
    fixed = read(PF/'model.json')
    assert fixed['fixed_weight'] == 1. and fixed['new_CLIP2_fitted_parameters'] == 0
    assert models['Y3']['weights'] == [0.2687633108971583], 'Do not replace original Y3 coefficient with LEAN-PF'
    assert coeff == {'ER01': 1., 'Y3': 1., 'Y2': .5, 'Y2-TF': 1., 'Y4': .25}
    assert fixed['external_prior_SHA'] == models['Z13']['external_prior_SHA']
    assert fixed['mean'] == models['Z13']['mean'] and fixed['std'] == models['Z13']['std']
    tr, va = (arr(V/f'evidence_relations_20261022/ER01/{s}.npz') for s in ['train', 'valid'])
    assert np.array_equal(tr['signal_ids'], z['train_signal_ids']) and np.array_equal(va['signal_ids'], z['valid_signal_ids'])
    rows = list(csv.reader((PF/'candidate/valid_predictions.csv').open(newline='', encoding='utf-8-sig')))
    ids = np.asarray([r[0] for r in rows])
    p = np.asarray([float(r[1])/100 for r in rows])
    assert len(ids) == len(set(ids)) == 2117 and np.array_equal(ids, z['valid_signal_ids'])
    assert sha(PF/'candidate/valid_predictions.csv') == 'a32b857ca9bf1da0bb4dcdcdf215bd4034ed74a7bfa060865b100d35b2209b74'
    valid, valid_stages = pipeline(z, models, coeff, 'valid', va['parent'].astype(float), fixed)
    max_score_error = float(np.max(np.abs(valid-p))*100)
    assert max_score_error < 1e-6, max_score_error
    prediction, stages = pipeline(z, models, coeff, 'train', tr['parent'].astype(float), fixed)
    assert len(prediction) == 3072 and np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    metadata = [json.loads(line) for line in (V/'fourth_round_20261023/Q0_Q5/train_metadata_linkage.jsonl').read_text(encoding='utf8').splitlines()]
    lookup = {r['signal_id']: (i, r) for i, r in enumerate(metadata)}
    assert len(lookup) == 16047 and all(str(s) in lookup for s in z['train_signal_ids'])
    mapped = [lookup[str(s)] for s in z['train_signal_ids']]
    npz_path = OUT/'PRIOR_FIXED_train_conditional.npz'
    np.savez_compressed(npz_path, signal_ids=z['train_signal_ids'], prediction=prediction, target=z['train_target'],
                        residual=z['train_target']-prediction, prediction_score=100*prediction,
                        target_score=100*z['train_target'], residual_score=100*(z['train_target']-prediction),
                        clean16047_index=np.asarray([i for i, r in mapped]),
                        audio2_sha256=np.asarray([r['audio2_sha256'] for i, r in mapped]),
                        lyric_sha256=np.asarray([r['lyric_sha256'] for i, r in mapped]),
                        leakage_proxy_group=np.asarray([r['leakage_proxy_group'] for i, r in mapped]),
                        legacy_lyric_fold=np.asarray([r['fold_lyrics_legacy'] for i, r in mapped]),
                        **{k+'_prediction': v for k, v in stages.items()})
    config = dict(source_sha256=source_sha, first_composition_coefficients=coeff,
                  third_composition_coefficients=frozen['third_composition_coefficients'],
                  node_models={k: models[k] for k in models}, final_PRIOR_FIXED_model=fixed,
                  algorithm_order=['587135 root', 'parallel Y2/Y2-TF/Y3/Y4 plus ER01 increments', 'ER02 minus ER01', 'Z2 power word node',
                                   'parallel Z8/Z11/Z9 word nodes', 'Z13 fixed standardized coefficient 1'],
                  standardization_clip=[-5, 5], sentence_logit_bound=.2, word_logit_bound=.2,
                  Y3_original_native_unit='empirical Bayes human-minus-machine probability residual',
                  Y3_coefficient_unchanged=models['Y3']['weights'], Z13_fixed_coefficient=1.,
                  fitted=False, candidate_generated=False, new_virtual_score_views=0)
    save(OUT/'frozen_node_configuration.json', config)
    limitations = [
        'This reproduces the exact frozen PRIOR_FIXED VALID graph and its matching 3072-row conditional TRAIN graph; it is not a newly cross-fitted model.',
        '3072 conditional TRAIN rows are a subset of 16047 cleaned TRAIN; no model predictions for the other 12975 rows were invented.',
        'Original MF/MOSS/P2 conditional mothers use legacy held lyric folds, which are not the proposed artist folds and differ from Q0 folds.',
        'Y3 lexical features used prompt cross-fitting, but its sentence coefficient was fitted on all3072; original ER01/ER02 replay uses full fitted heads on TRAIN.',
        'A/B ancestry and all upstream nuisance exclusion are unresolved; strict full joint OOF has not been established.',
        'ICC on these fixed conditional residuals is a structural diagnostic, not an estimate of artist-held-out predictive performance.',
        'A dual-protocol candidate replay must refit original candidate supervision on declared TRAIN folds with accurately matched conditional parents; simply slicing this vector into folds is not CV.',
        'PRIOR_FIXED is distinct from LEAN-PF: only Z13 fixed coefficient1 is its last-node mapping; Y3 remains .2687633108971583 and original composition remains frozen.'
    ]
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), status='verified_exact_frozen_graph',
                  source_code=str(Path(__file__)), source_code_SHA=sha(__file__),
                  frozen_prior_CSV=str(PF/'candidate/valid_predictions.csv'), frozen_prior_CSV_SHA=sha(PF/'candidate/valid_predictions.csv'),
                  VALID_identity_max_abs_score=max_score_error, VALID_labels_used=False, new_virtual_score_views=0,
                  TRAIN_rows=3072, clean_TRAIN_rows=16047, mapped_rows=len(mapped), all3072_unique=len(set(z['train_signal_ids'])) == 3072,
                  output=str(npz_path), output_SHA=sha(npz_path), residual_definition='target minus prediction, native0to1 and score0to100 supplied',
                  frozen_node_configuration=str(OUT/'frozen_node_configuration.json'), configuration_SHA=sha(OUT/'frozen_node_configuration.json'),
                  limitations=limitations, new_fit=False, strict_joint_OOF=False, LEAN_PF_predictions_used=False)
    save(complete, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
