"""Fit the repaired TRAIN-only external prior and stage one fixed own candidate.

No official submission, quota allocation, account mutation or existing winner
mutation is performed here. The six-slot reporting round is already complete.
"""
from v10_round3_common_20261023 import *
from scipy import sparse
import time
OUT = D / 'Z13_repair'
sha = T.sha


def prior():
    dest = OUT / 'external_word_prior.json'
    if dest.exists():
        return T.read(dest)
    q = T.read(OUT / 'replay_report.json')
    assert q['fraction'] >= .99 and q['rows'] == 8802
    rows = T.read(OUT / 'replayed_word_labels.json')
    rows = [r for r in rows.values() if r['hits_match']]
    wordlist = sorted({w for r in rows for w in r['reference'] if w != '?'})
    wi = {w: i for i, w in enumerate(wordlist)}
    si = {s: i for i, s in enumerate(sorted({r['content'] for r in rows}))}
    rr, cc, y = [], [], []
    for r in rows:
        sev = r['hearing_loss'].lower()
        sv = 2 if 'moderate' in sev else 1 if 'mild' in sev else 0
        for w, yy in zip(r['reference'], r['labels']):
            # Unknown reference slots are retained during replay/score checks,
            # but cannot define a transferable lexical effect.
            if w == '?':
                continue
            k = len(y)
            rr.extend([k, k, k]); cc.extend([0, 3+wi[w], 3+len(wi)+si[r['content']]])
            if sv:
                rr.append(k); cc.append(sv)
            y.append(float(yy))
    mat = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(y), 3+len(wi)+len(si)))
    y = np.asarray(y)
    penalty = np.r_[np.zeros(3), np.full(len(wi), 3.), np.full(len(si), 5.)]

    def fun(w):
        u = mat @ w
        return float(np.sum(np.logaddexp(0, u)-y*u)+.5*np.sum(penalty*w*w)), mat.T @ (expit(u)-y)+penalty*w

    fit = minimize(fun, np.zeros(mat.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=700, ftol=1e-10, gtol=1e-5))
    assert fit.success, str(fit.message)
    result = dict(UTC=T.utc(), word_effects={w: float(fit.x[3+i]) for w, i in wi.items()}, severity_coefficients=fit.x[:3].tolist(), word_parameters=len(wi), content_parameters=len(si), examples=len(rows), word_examples=len(y), excluded_ambiguous_rows=q['rows']-len(rows), blank_answers_retained=q['blank_answers'], CLIP1_TRAIN_only=True, VALID_or_EVAL_labels=False, prior='L2 logistic crossed word/content intercepts, penalties 3 and 5; no clipping or fitting to CLIP2 labels', convergence=dict(success=bool(fit.success), iterations=int(fit.nit), message=str(fit.message)), source_replay_SHA=sha(OUT/'replay_report.json'), source_labels_SHA=sha(OUT/'replayed_word_labels.json'))
    T.save(dest, result)
    return result


def candidate():
    out = OUT / 'current_own_candidate'
    if (out / 'ready.json').exists():
        print('Already staged; no resubmission', flush=True)
        return
    p = prior(); effects = p['word_effects']
    refpath = OUT / 'frozen_current_own.json'
    if not refpath.exists():
        T.save(refpath, T.read(T.V/'current_best_own_valid.json'))
    ref = T.read(refpath)
    tr, va, recs, tm, vm, tp, vp, folds = load()
    ids, valid_parent = G.F.csvpred(ref['csv'])
    assert np.array_equal(ids, va['signal_ids']) and ref['submission_id'] == 587265
    # TRAIN parent is the inherited, disclosed conditional surrogate. No
    # fabricated recursive OOF is claimed for the current winner.
    proxy = Z.word_proxy()
    vp = Z.anchor(proxy['valid_proxy'], vm, valid_parent)
    features, coverage = [], []
    for rows, mask in [(tr, tm), (va, vm)]:
        x = np.zeros((*mask.shape, 1)); covered = 0
        for i, sid in enumerate(rows['signal_ids']):
            for j, word in enumerate(recs[str(sid)]['reference']):
                x[i,j,0] = effects.get(word, 0.)
                covered += word in effects
        features.append(x); coverage.append(covered/int(mask.sum()))
    train, valid = features
    mean = train[tm].mean(0); std = np.maximum(train[tm].std(0), .01)
    a = np.clip((train-mean)/std, -5, 5); b = np.clip((valid-mean)/std, -5, 5)
    L = logit(np.clip(tp, 1e-9, 1-1e-9)); LV = logit(np.clip(vp, 1e-9, 1-1e-9))
    target = tr['target']; parent = tr['parent']; lam = .0002

    def predict(l, x, mask, w):
        return (expit(l+.2*np.tanh(x@w))*mask).sum(1)/mask.sum(1)

    def fit(use):
        x, l, mask, y = a[use], L[use], tm[use], target[use]
        n = mask.sum(1)
        def fun(w):
            v = np.tanh(x@w); q = expit(l+.2*v)
            error = (q*mask).sum(1)/n-y
            grad = np.einsum('nl,nlk->nk', .2*q*(1-q)*(1-v*v)*mask, x)/n[:,None]
            return float(np.mean(error*error)+lam*w@w), np.mean(2*error[:,None]*grad, 0)+2*lam*w
        f = minimize(fun, np.zeros(1), jac=True, method='L-BFGS-B', options=dict(maxiter=350, ftol=1e-13, gtol=1e-9))
        assert f.success, f.message
        return f.x

    oof = np.zeros(len(parent)); fold_weights = []
    for fold in range(5):
        w = fit(folds != fold); use = folds == fold; fold_weights.append(w.tolist())
        oof[use] = predict(L[use], a[use], tm[use], w)
    delta = oof-parent
    mse = {str(s): float(np.mean((parent+s*delta-target)**2)*10000) for s in [0., .5, 1.]}
    scale = float(min(mse, key=mse.get))
    w = fit(np.ones(len(parent), bool))
    raw = predict(LV, b, vm, w)
    pred = valid_parent+scale*(raw-valid_parent)
    zero = predict(LV, b, vm, np.zeros(1))
    zero_error = float(np.max(np.abs(zero-valid_parent)))
    order_error = float(np.max(np.abs(predict(LV[::-1], b[::-1], vm[::-1], w)[::-1]-raw)))
    assert zero_error < 1e-8 and order_error < 1e-12 and np.isfinite(pred).all() and pred.min() >= 0 and pred.max() <= 1
    out.mkdir(parents=True, exist_ok=True)
    model = dict(new_CLIP2_parameters=1, external_word_parameters=p['word_parameters'], external_content_parameters=p['content_parameters'], fields=['CLIP1_human_word_intercept'], mean=mean.tolist(), std=std.tolist(), weights=w.tolist(), scale=scale, fold_weights=fold_weights, word_logit_bound=.2, regularization=lam, external_prior_SHA=sha(OUT/'external_word_prior.json'), frozen_parent=ref)
    T.save(out/'model.json', model)
    np.savez_compressed(out/'features.npz', train=train, valid=valid, train_signal_ids=tr['signal_ids'], valid_signal_ids=va['signal_ids'])
    np.savez_compressed(out/'oof.npz', signal_ids=tr['signal_ids'], parent=parent, target=target, pred=oof, folds=folds)
    with (out/'valid_predictions.csv').open('w', newline='') as f:
        csv.writer(f).writerows(zip(ids, [format(float(v)*100,'.10f') for v in pred]))
    report = dict(UTC=T.utc(), parent_submission=ref['submission_id'], frozen_parent_RMSE=ref['RMSE'], rows=len(ids), coverage=dict(TRAIN=coverage[0], VALID=coverage[1]), scale=scale, OOF_MSE_by_scale=mse, conditional_OOF_gain_MSE=mse['0.0']-mse[str(scale)], zero_coefficient_replay_max_abs=zero_error, row_order_replay_max_abs=order_error, delta_RMS_score=float(np.sqrt(np.mean((pred-valid_parent)**2))*100), csv_SHA=sha(out/'valid_predictions.csv'), model_SHA=sha(out/'model.json'), TRAIN_parent='Inherited conditional 3072-row surrogate, not recursively recomputed OOF of current complete winner', selection='Fixed 0/half/full TRAIN prompt-group rule; no VALID labels', gate_replay_fraction=T.read(OUT/'replay_report.json')['fraction'], official_submitted=False, official_gate='Existing six-slot round already completed; does not allocate or create another submission')
    T.save(out/'report.json', report)
    T.save(out/'verification.json', dict(status='verified', csv_sha256=report['csv_SHA'], report_sha256=sha(out/'report.json')))
    T.save(out/'ready.json', dict(UTC=T.utc(), scale=scale, candidate=str(out), official_submitted=False))
    progress('Z13', f'重放99.364%通过；外部真人词项先验和1参数头已完成，固定幅度{scale:g}；未提交官测', '候选已就绪；原6次预算已用完，保留下一可用官测窗口')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    candidate()
