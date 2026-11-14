"""Record user-supplied official screenshots, summarize round 3, stage controls."""
from v10_round3_common_20261023 import *
import shutil

B = D / 'Z13_attribution_20261023'
R = D / 'Z13_repair'
NOT_BEFORE = '2026-10-24T00:00:00+00:00'
DEADLINE_AT = '2026-10-24T04:00:00+00:00'


def manual_results():
    p = B/'user_reported_official.json'
    if p.exists():
        return T.read(p)
    B.mkdir(exist_ok=True)
    manifest = T.read(R/'manual_test_exports/manifest.json')
    supplied = [
        ('own', Path(r'C:\Users\zjh13\Documents\WeChat Files\wxid_kuwmschtlqo312\FileStorage\Temp\8970853bcedda395588fc21e19d2eb9.png'), 15.508941976315871, .8722210806782259),
        ('joint', Path(r'C:\Users\zjh13\Documents\WeChat Files\wxid_kuwmschtlqo312\FileStorage\Temp\33c740cc142b66d9a61cba778dc8edf.png'), 15.304203078842406, .8757444323908927),
    ]
    rows = {}
    for kind, source, rmse, corr in supplied:
        target = B/(kind+'_user_result.png')
        shutil.copyfile(source, target)
        source_csv = Path(manifest[kind]['csv'])
        if not source_csv.exists():
            # The user renamed the exports to 01.csv and 02.csv to upload.
            matches=[f for f in (R/'manual_test_exports').glob('*.csv') if T.sha(f)==manifest[kind]['SHA']]
            assert len(matches)==1, (kind,'Cannot uniquely bind renamed user export by SHA')
            source_csv=matches[0]
        assert T.sha(source_csv) == manifest[kind]['SHA']
        rows[kind] = dict(UTC=T.utc(), submission_id=None, RMSE=rmse, Correlation=corr,
            result=dict(RMSE=rmse, Correlation=corr), csv=str(source_csv), csv_SHA=T.sha(source_csv),
            score_source='User-supplied screenshot of official result; submission ID not supplied',
            screenshot=str(target), screenshot_SHA=T.sha(target), server_input_independently_verified=False,
            mapping_basis='User returned two results after the two named CSV exports; own/joint order and metric levels agree',
            latest_task='Z13_USER_REPORTED', full_raw_infer_all_complete=False)
    result = dict(UTC=T.utc(), **rows, handoff_manifest=str(R/'manual_test_exports/manifest.json'),
        own_parent_RMSE=15.525338913669483, joint_parent_RMSE=15.31426593870339,
        own_gain=15.525338913669483-rows['own']['RMSE'], joint_gain=15.31426593870339-rows['joint']['RMSE'],
        official_receipt_not_fabricated=True)
    T.save(p, result)
    for kind in ['own','joint']:
        path = T.V/('current_best_own_valid.json' if kind=='own' else 'current_best_joint_valid.json')
        old = T.read(path)
        score = old['RMSE'] if kind=='own' else old['result']['RMSE']
        if rows[kind]['RMSE'] < score:
            T.save(path, dict(rows[kind], previous_best=old, evidence_path=str(p)))
    T.save(B/'Z13_manual_completion.json', dict(UTC=T.utc(), completed='Repaired scoring, external human lexical prior, tiny readout, and user-reported own/joint official tests',
        evidence=str(p), evidence_SHA=T.sha(p), own_gain=result['own_gain'], joint_gain=result['joint_gain'],
        screenshot_results_accepted_as_user_reported=True, submission_IDs_pending=True))
    return result


def emit(tid, pred, model, extra, ids, parent):
    c=B/tid/'candidate';c.mkdir(parents=True,exist_ok=True)
    if (c/'verification.json').exists():
        return
    assert pred.shape==(2117,) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    T.save(c.parent/'model.json',model)
    with (c/'valid_predictions.csv').open('w',newline='') as f:
        csv.writer(f).writerows(zip(ids,[format(float(v)*100,'.10f') for v in pred]))
    report=dict(UTC=T.utc(),task=tid,csv_SHA=T.sha(c/'valid_predictions.csv'),model_SHA=T.sha(c.parent/'model.json'),
        frozen_parent=parent,rows=2117,VALID_labels=False,EVAL=False,TRAIN_score_submission_gate=False,
        scope='One prespecified counterfactual; no TRAIN holdout gate and no VALID coefficient search',extra=extra)
    T.save(c/'report.json',report)
    T.save(c/'verification.json',dict(status='verified',csv_sha256=report['csv_SHA'],report_sha256=T.sha(c/'report.json')))


def prepare_controls():
    if (B/'controls_ready.json').exists():
        return
    tr,va,recs,tm,vm,tp,vp,fold=load()
    parent=T.read(R/'frozen_current_own.json')
    ids,p=G.F.csvpred(parent['csv'])
    assert parent['submission_id']==587265 and np.array_equal(ids,va['signal_ids'])
    proxy=Z.word_proxy();vp=Z.anchor(proxy['valid_proxy'],vm,p)
    L=logit(np.clip(tp,1e-9,1-1e-9));LV=logit(np.clip(vp,1e-9,1-1e-9))
    def predict(l,mask,w):
        return (expit(l+.2*np.tanh(w))*mask).sum(1)/mask.sum(1)
    def fun(w):
        t=np.tanh(w[0]);q=expit(L+.2*t);error=(q*tm).sum(1)/tm.sum(1)-tr['target']
        gradient=(.2*(1-t*t)*q*(1-q)*tm).sum(1)/tm.sum(1)
        return float(np.mean(error*error)+.0002*w@w),np.array([np.mean(2*error*gradient)+.0004*w[0]])
    fit=minimize(fun,np.zeros(1),jac=True,method='L-BFGS-B',options=dict(maxiter=350,ftol=1e-13,gtol=1e-9))
    assert fit.success,fit.message
    head=predict(LV,vm,fit.x[0]);zero=predict(LV,vm,0.)
    assert np.max(abs(zero-p))<1e-8 and np.max(abs(head-predict(LV[::-1],vm[::-1],fit.x[0])[::-1]))<1e-12
    emit('HEAD_ONLY',head,dict(new_CLIP2_parameters=1,fields=['constant_word_logit_calibration'],weight=float(fit.x[0]),word_logit_bound=.2,regularization=.0002),
        dict(question='Can a matched one-parameter calibration improve without any CLIP1 information?',zero_replay=float(np.max(abs(zero-p))),
             inherited_conditional_TRAIN_parent=True,no_external_word_feature=True),ids,parent)
    features=G.P.F.arr(R/'current_own_candidate/features.npz')
    assert np.array_equal(features['valid_signal_ids'],ids) and np.array_equal(features['train_signal_ids'],tr['signal_ids'])
    original=T.read(R/'current_own_candidate/model.json')
    b=np.clip((features['valid']-np.asarray(original['mean']))/np.asarray(original['std']),-5,5)
    assert b.shape==(*vm.shape,1)
    # A fixed +1 coefficient on the identical standardized external feature.
    # This deliberately omits fitting its CLIP2 coefficient (full Z13 fitted .51846).
    fixed=(expit(LV+.2*np.tanh(b[...,0]))*vm).sum(1)/vm.sum(1)
    reverse=(expit(LV[::-1]+.2*np.tanh(b[::-1,...,0]))*vm[::-1]).sum(1)/vm[::-1].sum(1)
    assert np.max(abs(reverse[::-1]-fixed))<1e-12
    emit('PRIOR_FIXED',fixed,dict(new_CLIP2_fitted_parameters=0,fixed_weight=1.,word_logit_bound=.2,mean=original['mean'],std=original['std'],external_prior_SHA=original['external_prior_SHA']),
        dict(question='Does the correctly mapped external prior help under a prespecified fixed readout, without fitting a CLIP2 coefficient?',
             comparison_full_Z13_fitted_weight=original['weights'][0],limitation='This tests a specific fixed mapping; no unique head-free predictor exists'),ids,parent)
    T.save(B/'score_repair_only_identity.json',dict(UTC=T.utc(),scope='Repair changed CLIP1 label reconstruction only; no changes to the main 587265 prediction path',
        prediction_csv=str(Path(parent['csv'])),SHA=T.sha(Path(parent['csv'])),maximum_prediction_change=0,official_trial='Skipped: byte-identical baseline, consumes no quota'))
    T.save(B/'controls_ready.json',dict(UTC=T.utc(),candidates=['HEAD_ONLY','PRIOR_FIXED'],TRAIN_score_gate=False,full_Z13_resubmitted=False))


def register_queue():
    if (B/'execution_protocol.json').exists():
        return
    F.submit_candidates([F.candidate(B/t/'candidate','v10_Z13_ATTR_'+t+'_20261023','v10_Z13_ATTR_'+t+'_20261023','Prespecified Z13 attribution control '+t) for t in ['HEAD_ONLY','PRIOR_FIXED']],B/'official')
    spec=T.read(B/'official/manifest.json')
    known={r['csv_sha256']:r for r in spec.get('known_submissions',[])}
    for row in T.read(D/'official_registry.json')['candidates']:
        r=T.read(Path(row['candidate'])/'official_result.json');known[r['csv_SHA']]=dict(csv_sha256=r['csv_SHA'],submission_id=r['submission_id'])
    for row in T.read(B/'user_reported_official.json').values():
        if isinstance(row,dict) and row.get('csv_SHA'):
            known[row['csv_SHA']]=dict(csv_sha256=row['csv_SHA'],submission_id=None,user_manually_reported=True)
    spec['known_submissions']=list(known.values());spec['reference_csv']=str(R/'frozen_current_own.json')
    spec['reference_csv']=T.read(R/'frozen_current_own.json')['csv']
    spec['submission_policy']=dict(effective_at_utc=NOT_BEFORE,max_new_per_day=4,max_concurrent=4,reserve_daily_slots=1,min_submission_interval_seconds=120,require_explicit_selection=True)
    T.save(B/'official/manifest.json',spec)
    T.save(B/'candidate_registry.json',dict(candidates=[dict(id=c['id'],task=t,kind='own',candidate=str(B/t/'candidate'),selected=True) for t,c in zip(['HEAD_ONLY','PRIOR_FIXED'],spec['candidates'])]))
    protocol=dict(UTC=T.utc(),user_request='Summarize third round; attribution tests after quota recovery; retain official-positive changes',
        not_before_UTC=NOT_BEFORE,deadline_UTC=DEADLINE_AT,max_own_attempts=3,max_total_attempts=4,
        initial_controls=['HEAD_ONLY','PRIOR_FIXED'],no_REPEAT_manual_Z13_or_joint=True,TRAIN_holdout_gate=False,
        primary_comparison='Common frozen 587265 baseline vs head-only / prior-fixed / already measured full Z13',
        conditional_composition='At most one; only if HEAD_ONLY is official-positive vs587265, fixed half for correlated increments, otherwise full; algebra forecast must beat current best by0.0005',
        conditional_joint='At most one after own tests; require own gain>=0.01 over user-reported15.508942 and algebra forecast improvement>=0.0005 over joint15.304203',
        adoption='Actual own official results decide; no newly fitted VALID coefficients; negative unchanged or rolled back',
        score_mapping_evidence='Screenshots do not include submission IDs; manual results retained explicitly as user-reported',
        needs_no_GPU=True,one_finite_worker=True)
    T.save(B/'execution_protocol.json',protocol)


def write_summary():
    manual=T.read(B/'user_reported_official.json')
    base=T.read(D/'own_reference.json');ref=15.525338913669483
    lines=['# 第三轮总结与Z13收益拆分','',
        '本轮起点：本人15.551907842（587241）；上一轮联合15.358069353。',
        f'最新截图反馈：本人 **{manual["own"]["RMSE"]:.9f}**，联合 **{manual["joint"]["RMSE"]:.9f}**。截图缺少提交号；按用户返回的本人/联合顺序绑定此前导出的CSV和SHA，保留为用户回报的官方结果，不虚构API回执。','',
        '## 一、第三轮逐项结果','',
        '|任务|实际做法|官测RMSE|相对共同起点的收益|结论|',
        '|---|---|---:|---:|---|']
    names={'Z6':'熟悉度、习得年龄、具体性等常模，6参数','Z8':'同一掩码语言模型的竞争词支持余量＋整句PLL，2参数','Z11':'词尾形态×轻/中度听损，6参数','Z9':'PANNs分耳Audio2场景概率，6字段'}
    for tid in ['Z6','Z8','Z11','Z9']:
        r=T.read(D/tid/'candidate/official_result.json');score=r['result']['RMSE'];gain=base['RMSE']-score
        lines.append(f'|{tid}|{names[tid]}|{score:.9f}（{r["submission_id"]}）|{gain:+.6f}|'+('未采用' if gain<0 else '已保留；Z11幅度很小，仍需条件贡献检查' if tid=='Z11' else '已保留')+'|')
    lines += [f'|三个正向修正固定合成|Z8＋Z11＋Z9相对同一母方案的增量，未连续搜索权重|15.525338914（587265）|{base["RMSE"]-ref:+.6f}|采用|',
        f'|Z13完整接入|修复CLIP1判词→真人词项先验→1参数有界读出|{manual["own"]["RMSE"]:.9f}（截图）|相对587265再降{manual["own_gain"]:.6f}|采用完整接入，内部归因待测|',
        '|Z7 记忆与词序|位置×时长负荷及对照|未官测|无法认定官方收益|首轮固定幅度为0，不能写成官测失败|',
        '|Z12 共识误听|MF/MOSS/Whisper缓存的对齐错词一致性|未官测|无法认定官方收益|首轮固定幅度为0，不能写成官测失败|','',
        '本人本轮合计降低 **'+format(base['RMSE']-manual['own']['RMSE'],'.6f')+'** RMSE。联合由15.358069353→15.314265939（587266）→'+format(manual['joint']['RMSE'],'.9f')+'，本轮合计降低 **'+format(15.358069352757893-manual['joint']['RMSE'],'.6f')+'**。',
        '各单项数字来自对共同母方案的比较，不能当作加入当前最终系统后的独立贡献直接相加。Z8是已测三项中最大的单项；Z11的0.000366非常小，不足以单独证明词尾听损机制。','',
        '## 二、Z13的三个环节分别做了什么','',
        '|环节|作用|目前可以确认什么|',
        '|---|---|---|',
        '|计分重放修复|从原人类回答生成合法缩写/同音替代表达，保留?词位，修复CLIP1训练词标签|一致率由8334/8802＝94.683%升至8746/8802＝99.364%；只是外部监督的前置修复，没有改主方案输出。单独修复的预测与587265相同，不另交一次官测。|',
        '|外部真人词项先验|在8746条真实TRAIN回答上估计3130个词效应、2931个内容截距，并控制听损|新增了此前模型没有见过的真实人类词项信息；保留1339条空答。56条词计分仍不一致，仅不用于词标签拟合，未删除源数据。|',
        '|新评分头|把外部词效应转换为当前系统的轻微逐词logit修正|只增加1个CLIP2系数，权重约0.51846、有界0.2；没有重训整个C或上游大听者。它与先验共同产生此次0.016397收益，现有两份成绩不能给二者分别记功。|','',
        '## 三、额度恢复后的匹配测试','',
        '|候选|保留什么|检验的问题|新增额度|',
        '|---|---|---|---:|',
        '|母方案587265|无Z13|共同对照，已有成绩15.525338914|0|',
        '|只修计分|不接入外部先验|预测不变，机械核验即可|0|',
        '|HEAD_ONLY|无CLIP1输入，只拟合同容量1参数有界全局词logit校准|收益是否仅来自再校准原模型|1|',
        '|PRIOR_FIXED|同一外部词特征，固定标准化系数+1，不拟合CLIP2系数|固定读出下外部信息是否仍有效；和完整版本比较拟合系数的作用|1|',
        '|完整Z13|外部先验＋已拟合1参数头|已经出分15.508941976，不重复提交|0|',
        '|条件合成|若HEAD_ONLY正向且固定合成有代数增益，最多再交1份|避免把重叠的正向项机械叠加|≤1|',
        '|条件联合|本人比15.508942再降至少0.01才考虑，且联合预测有增量|节省联合额度，优先本人指标|≤1|','',
        '两个主要对照已生成。固定读出是一种明确约定，不是唯一“没有头”的实现，不能把两次对照的差额解释成普遍、精确的机制贡献。保留依据仍是实际官测；不以训练留出分数拦截。',
        '禁止使用未达99%资格的旧外部词标签再训练一个“坏标签”模型来消耗官测。计分修复的必要性由测量完整性确认，不把代码修复包装成一个独立提分模块。','',
        '## 四、执行时间与采用规则','',
        '5个已登记账号在本轮查询时剩余额度均为0。后台单次有限队列从 **2026-10-24 00:00 UTC** 开始重查真实额度，即 **北京时间10月24日08:00 / 洛杉矶10月23日17:00**。',
        '预计首批2份在恢复后20–60分钟得到结果；额外合成/联合依赖前面的实际结果，预算上限4次，不保证全部使用。网络或平台排队会延长；有限队列到10月24日04:00 UTC停止新提交。',
        '已测完整Z13先保留。对照若不胜当前本人最佳，不替换；若其修正有条件增量，只在新合成官测通过后并入。负向回退到已保存最佳。既有两份手动CSV不重复提交。','',
        '## 五、边界与产物','',
        '- 全部是VALID探索结果；截图分数与API核验回执分别保存，提交号待补。',
        '- 新评分头TRAIN父预测仍是既有3072条条件代理，未补做递归联合OOF；这不是本轮新增长训练任务。',
        '- 此次归因无需GPU，复用缓存即可；不重跑已完成的大听者或已确认无效的配方。',
        '- `user_reported_official.json`与两张图片：最新成绩证据。',
        '- `HEAD_ONLY/candidate/`、`PRIOR_FIXED/candidate/`：固定候选、模型与核验。',
        '- `execution_protocol.json`、`official/manifest.json`：等待恢复的有限预算队列。',
        '- `runner_state.json`、`official/receipts/`、`official_results.json`：后台状态与实际回执。']
    T.write(B/'第三轮总结与Z13归因测试.md','\n'.join(lines)+'\n')


def tasks():
    with T.locked() as ok:
        assert ok
        state=T.read(T.STATE)
        if 'Z13-ATTR' not in {r['id'] for r in state['tasks']}:
            state['tasks'].append(dict(id='Z13-ATTR',priority='P0',task='Z13新增真人先验与1参数读出的匹配官测归因，保留正向',
                state='HEAD_ONLY、PRIOR_FIXED两份固定候选已就绪；等待UTC额度恢复，旧完整Z13不重复提交',
                ETA='最快北京时间10月24日08:00恢复后20–60分钟首批；条件合成/联合另计',
                dependency='自动有限队列：2次主要对照＋最多1次有依据的合成＋最多1次有明显本人增量的联合；最多4次',
                protocol=str(B/'execution_protocol.json'),source_id='用户20261023要求第三轮总结和拆分归因'))
        T.save(T.STATE,state)
    if 'Z13' in {r['id'] for r in T.read(T.STATE)['tasks']}:
        T.mark_complete('Z13',[str(B/'Z13_manual_completion.json'),str(B/'user_reported_official.json')])
    T.refresh()


def main():
    manual_results();prepare_controls();register_queue();write_summary();tasks()
    print(json.dumps(dict(status='prepared',summary=str(B/'第三轮总结与Z13归因测试.md'),not_before=NOT_BEFORE),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
