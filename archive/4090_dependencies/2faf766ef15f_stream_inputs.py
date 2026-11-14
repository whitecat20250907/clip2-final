"""Fixed FIT-only transfer rows, verified frame sources and own D6e trunk."""
import ast
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from grounding import Matcher
from transfer_core import normalized_words, scoring_mapping

HERE = Path(__file__).resolve().parent
EXPS = HERE.parent
DIAGNOSTIC = EXPS/'d6e_phonetic_grounding_diagnostic_v1'
OLD = EXPS/'d6e_phonetic_grounding_v1'
MATCHER_PROTOCOL = '9228cc7314865684545ba077768c6915caea21374a3510d39ea6922c50c9765d'
CHECKPOINTS = {
    'true_pair': 'cd33290ddb3d68acac0557fc67da333a939f1b81fcd46c041752f8c6ec08b3e8',
    'wrong_positive': 'e71c41b7204bcc06b306f03d25bfcf83279e29f8303c1a706c9aa5ca41a6fb68',
}
EXPECTED_INITIAL = '884690fada98ee9a0b889045b07be92659baaec5f88a56c3574a923c7c0f8a9c'
EXPECTED = {
    'train_ids': '4c2a30ff850d55fb1befc2e4c0d9f32002dd2de647a4387338204964534f9b7a',
    'dev_ids': '6c486a943337877860c67ac871194c84481944106d54e057c15ab556e96149a3',
    'excluded_train_ids': 'f49c3aadc9030817d93b8ca485617a6d4e179cc45cb98e12d4978bd3a261db2e',
    'fit_ids': '012dccfa168f824137814af89bfb8168f86bcac3f4e6b1487f8a210c15bc6078',
    'inner_ids': '3ddd15ecd2ab4aca714200b43e56b3ac0fee52a062701829520ec2f642533a89',
}
SIDE = ('confidence_v3','confidence_v2','wer','extra_wer','candidate','paired')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def helper():
    path = OLD/'probe_inputs.py'
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != '59ad9a41749346098480bce1ede4a42d13cb9481d085ab288e81bc89fe904e46':
        raise ValueError('Audited public input helper changed')
    return module(path,'_grounded_transfer_public_helper')


def state_hash(state):
    import hashlib
    h = hashlib.sha256()
    for name, x in sorted(state.items()):
        a = x.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def query(row, public, queries, dictionary, contractions, m):
    """A failed auxiliary query masks its word branch; never drop score rows."""
    words = normalized_words(row['original_prompt'])
    target_words = public['words']
    word_map = scoring_mapping(words,target_words,contractions)
    try:
        tokens = queries[row['query_id']]['tokens']
        m.require(bool(tokens),'Unresolved recorded sung-phone query')
        checked_words, parts = m.blocks(row['original_prompt'],dictionary,tokens)
        m.require(checked_words == words,'Sung word normalization differs')
        ids, starts = m.text_arrays(parts)
        m.require(len(ids) <= math.ceil(row['files']['audio2']['seconds']/.02),'Too few Audio2 frames')
        belongs = [word_map[i] for i, block in enumerate(parts) for _ in block]
        return dict(phones=ids,boundaries=starts,phone_to_word=belongs,reason=None,
                    mapped_words=len(set(x for x in belongs if x >= 0)),sung_words=len(words),scoring_words=len(target_words))
    except ValueError as e:
        return dict(phones=[],boundaries=[],phone_to_word=[],reason=str(e),mapped_words=0,
                    sung_words=len(words),scoring_words=len(target_words))


def prepare(args):
    m = helper(); root = args.assets.resolve(); p = m.read(root/'protocol.json')
    partition = p['partition']
    m.require({k:m.digest(v) for k,v in partition.items()} == EXPECTED,'Canonical cohort/order/exclusions changed')
    m.require(not set(partition['train_ids']) & set(partition['dev_ids'])
              and set(partition['fit_ids']).isdisjoint(partition['inner_ids'])
              and set(partition['fit_ids'])|set(partition['inner_ids']) == set(partition['train_ids']), 'Partition overlap')
    public = m.read(root/'public.json'); m.require(m.digest(public) == p['public_sha256'],'Public inputs changed')
    gp = Path(p['readiness'])/'grouping.json'
    m.require(m.sha(gp) == p['grouping_sha256'],'Grouping changed')
    grouping = m.read(gp)
    m.require(set(grouping['excluded_ids'])-set(partition['dev_ids']) == set(partition['excluded_train_ids']), 'Actual34 exclusions changed')
    group_of = {s:m.digest(sorted(g['signals'])) for g in grouping['groups'] for s in g['signals']}
    plan_path = Path(p['source'])/'results/clip2_full_v1/plan.json'
    m.require(m.sha(plan_path) == p['plan_sha256'],'Plan changed')
    plan = m.read(plan_path); m.require(plan['dev_ids'] == partition['dev_ids'],'Original DEV order changed')
    rows = {r['signal']:r for r in plan['rows']}
    report = m.read(args.matching/'report.json'); protocol = m.read(args.matching/'protocol.json')
    m.require(report['protocol_sha256'] == MATCHER_PROTOCOL == m.digest(protocol)
              and report['matching_resource_gate_passed'] is True
              and report['held_used_for_checkpoint_selection'] is False,'Wrong/incomplete matching diagnostic')
    input_audit = m.read(args.matching/'input_audit.json')
    m.require(m.sha(args.matching/'input_audit.json') == protocol['input_audit_sha256'],'Matching input audit changed')
    m.require(input_audit['source_assets_protocol_sha256'] == m.sha(root/'protocol.json'),'Matching source differs')
    m.require(m.digest([r['signal'] for r in input_audit['train_rows']]) == report['train_ids_sha256']
              and m.digest([r['signal'] for r in input_audit['hold_rows']]) == report['held_ids_sha256'],'Matching cohort identity changed')
    excluded = {r['group'] for r in input_audit['train_rows']+input_audit['hold_rows']}
    excluded.update(group_of[s] for s in input_audit['excluded_probe_ids'])
    chosen, seen = [], set()
    for sid in sorted(partition['fit_ids'],key=lambda s:m.digest(['grounded-transfer-probe-v1',s])):
        group = group_of[sid]
        if group in excluded or group in seen:
            continue
        seen.add(group); chosen.append(sid)
        if len(chosen) == 32:
            break
    m.require(len(chosen) == 32,'Need32 fixed new FIT groups; no cache/label-based replacement')
    dpath = args.dictionary
    if not dpath.exists(): dpath = dpath.parent/'transfer_lexicon/cmudict.dict'
    dictionary = m.lexicon(dpath)
    qs, counts = {}, {}
    for phase in ('train_ids','dev_ids'):
        total, mapped, unresolved, complete = 0,0,0,0
        for sid in partition[phase]:
            m.require(rows[sid]['original_prompt'] == public[sid]['original_prompt'],'Sung text differs')
            try:
                q = query(rows[sid],public[sid],plan['queries'],dictionary,p['contractions'],m)
            except ValueError as e:
                q = dict(phones=[],boundaries=[],phone_to_word=[],reason=str(e),mapped_words=0,
                         sung_words=None,scoring_words=public[sid]['n_words'])
            qs[sid] = q
            total += q['scoring_words']; mapped += q['mapped_words']; unresolved += bool(q['reason'])
            complete += q['mapped_words'] == q['scoring_words']
        counts[phase] = dict(rows=len(partition[phase]),scoring_words=total,mapped_words=mapped,
                            masked_words=total-mapped,unresolved_query_rows=unresolved,all_words_supported_rows=complete)
    m.require(all(s not in partition['dev_ids'] for s in chosen),'Probe touches DEV')
    audit = dict(schema='paired_grounding_transfer_inputs_v1',selected_ids=chosen,
                 selected_ids_sha256=m.digest(chosen),selection='32 distinct SHA-selected FIT groups outside previous32+512+128 matching groups',
                 selected_by_cache_presence=False,selected_by_target=False,retained_train=16204,original_dev=4059,
                 id_hashes=EXPECTED,mapping=counts,selected_queries={s:qs[s] for s in chosen},
                 contractions_without_phone_boundaries='mask one-to-many expanded sung-word blocks',
                 source_assets_protocol_sha256=m.sha(root/'protocol.json'),source_plan_sha256=p['plan_sha256'],
                 matching_protocol_sha256=MATCHER_PROTOCOL,source_metadata_parsed_in_full=False,
                 public_plan_and_mapping_parsed_in_full=True,regression_scores_read=False,listener_responses_read=False)
    return m,p,public,rows,qs,chosen,audit


def final_frames(args, ids, rows, m):
    final = m.read(args.cache/'protocol.json')
    prefix = m.read(args.cache.parent/'phone_memory_prefix_cache_v1/protocol.json')
    m.require(final['schema'] == 'phone_memory_valid_frozen_l32_v1' and final['encoder_frozen']
              and final['trim_after_full_context'] and final['labels_used'] is False,'Wrong Audio2 cache')
    m.require(m.digest(prefix) == final['prefix_protocol_sha256'],'Final/prefix lineage differs')
    owner = args.cache.parent.parent
    m.require(m.sha(owner/'frozen_runtime.py') == final['implementation_sha256'],'Final cache code changed')
    for name,h in prefix['code_sha256'].items():
        m.require(Path(name).name == name and m.sha(owner/name) == h,'Prefix source changed')
    f, commits, missing = {}, [], []
    for sid in ids:
        info = rows[sid]['files']['audio2']; key = m.digest(['full-context-prefix30-v1',info['pcm_sha256'],0])
        path = args.cache/key[:2]/(key+'.npz'); mp = path.with_suffix('.json')
        if not path.exists() or not mp.exists():
            missing.append(dict(signal=sid,key=key)); continue
        c = m.read(mp); st = path.stat()
        m.require(c['protocol_sha256'] == m.digest(final) and c['kind'] == 'frames'
                  and c['stat'] == [st.st_size,st.st_mtime_ns] and m.sha(path) == c['sha256'],'Audio2 cache changed')
        with np.load(path,allow_pickle=False) as z:
            m.require(z.files == ['frames'],'Wrong Audio2 cache fields'); x = z['frames'].copy()
        m.require(x.dtype == np.float16 and x.shape == (math.ceil(info['seconds']/.02),1280)
                  and np.isfinite(x).all(),'Invalid Audio2 frame shape')
        f[sid] = x.astype(np.float32); commits.append(dict(signal=sid,sha256=c['sha256'],path=str(path)))
    return f,dict(missing=missing,entries=commits,protocol_sha256=m.digest(final),plan_sha256=prefix['plan_sha256'],
                  feature_protocol_sha256=prefix['feature_protocol_sha256'])


def baseline_and_audio1(args, p, ids, public, m, device):
    previous = args.reference.resolve(); source = args.assets.parent.parent.resolve()
    m.require(source in previous.parents,'Reference must be own completed multitrack experiment')
    old = m.read(previous/'report.json'); old_protocol = m.read(previous/'protocol.json')
    m.require(old['status'] == 'full_train_original_4059_dev_comparison_complete' and old['id_hashes'] == EXPECTED
              and old['protocol_sha256'] == m.digest(old_protocol)
              and old_protocol['assets_protocol_sha256'] == m.digest(p),'Completed reference lineage changed')
    for name,h in p['code_sha256'].items():
        m.require(Path(name).name == name and m.sha(source/name) == h,'Own multitrack source changed')
    rawpath = args.assets/'raw.npz'
    m.require(m.sha(rawpath) == old_protocol['raw_sha256'] == m.read(args.assets/'raw_manifest.json')['sha256'],'Raw source changed')
    with np.load(rawpath,allow_pickle=False) as z:
        signals = z['signals'].tolist()
        m.require(signals == p['partition']['train_ids']+p['partition']['dev_ids'],'Raw order changed')
        index = {s:i for i,s in enumerate(signals)}; take = [index[s] for s in ids]
        raw = {k:z[k][take].copy() for k in ('pooled',)+SIDE}
    done = m.read(previous/'matched_reference_select_complete.json')
    cp = previous/done['checkpoint']; npth = previous/'selection_normalization.json'
    m.require(cp.resolve().parent == previous and done['phase'] == 'select' and done['arm'] == 'matched_reference'
              and done['status'] == 'complete' and done['protocol_sha256'] == old['protocol_sha256']
              and done['train_ids_sha256'] == EXPECTED['fit_ids']
              and m.sha(cp) == done['checkpoint_sha256'] and m.sha(npth) == old['normalization_sha256'][npth.name], 'Selection baseline changed')
    norm = m.read(npth)
    m.require(norm['fit_ids_sha256'] == EXPECTED['fit_ids'] and norm['fit_n'] == 14515,'Baseline normalization touches INNER/DEV')
    parts = []
    for k in SIDE:
        x = raw[k].astype(np.float32)
        if k in norm['side']:
            v = norm['side'][k]; x = (x-np.asarray(v['mean'],np.float32))/np.asarray(v['std'],np.float32)
        parts.append(x)
    side = np.concatenate(parts,-1)
    # Only the hash-verified own ReferenceModel class; no trainer/main execution.
    tree = ast.parse((source/'model.py').read_text(encoding='utf-8-sig'))
    node = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'ReferenceModel']
    m.require(len(node) == 1,'Own reference architecture missing')
    scope = dict(nn=nn,torch=torch)
    exec(compile(ast.Module(body=node,type_ignores=[]),str(source/'model.py'),'exec'),scope)
    model = scope['ReferenceModel'](1280,side.shape[1]).to(device)
    ck = torch.load(cp,map_location='cpu',weights_only=True)
    m.require(ck['phase'] == 'select' and ck['arm'] == 'matched_reference'
              and ck['protocol_sha256'] == old['protocol_sha256'],'Wrong reference checkpoint')
    model.load_state_dict(ck['weights']); model.requires_grad_(False).eval()
    with torch.no_grad():
        pooled = torch.tensor(raw['pooled'],device=device); sv = torch.tensor(side,device=device)
        logits = model.head[:-1](torch.cat((model.projection(pooled),sv),-1)).squeeze(-1)
        gap = float((logits.sigmoid()-model(pooled,sv)).abs().max())
    m.require(gap < 1e-6,'Own frozen baseline forward mismatch')
    lib = Path(p['cache_loader'])
    m.require(m.sha(lib) == p['cache_loader_sha256'],'Historical Audio1 loader changed')
    if tuple(lib.parts[-6:]) == ('clarity','recipes','clip2','stage1_20260811','common','cache.py'):
        project = str(lib.parents[5])
        if project not in sys.path: sys.path.insert(0,project)
        mod = importlib.import_module('clarity.recipes.clip2.stage1_20260811.common.cache')
        m.require(Path(mod.__file__).resolve() == lib.resolve(),'Wrong installed cache module')
    else:
        mod = module(lib,'_grounded_verified_legacy_cache')
    heard, records = {}, []
    for i,sid in enumerate(ids):
        x = mod.load_cached_layers(str(p['cache_root']),sid,cache_format='npy_single',feature_dim=1280,requested_layers=[-1])[-1]
        if hasattr(x,'detach'): x = x.detach().cpu().float().numpy()
        x = np.asarray(x,np.float32)
        m.require(x.shape == (1500,1280) and np.isfinite(x).all(),'Invalid original50Hz Audio1 cache')
        pooled_gap = float(abs(x.mean(0)-raw['pooled'][i]).max())
        m.require(pooled_gap <= 1e-4,'Original Audio1 cannot replay frozen D6e pooled features')
        n = math.ceil(public[sid]['files']['audio1']['seconds']/.02)
        m.require(0 < n <= 1500,'Audio1 duration out of range')
        heard[sid] = x[:n].copy()
        import hashlib
        records.append(dict(signal=sid,full_frame_content_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                            baseline_pooled_max_gap=pooled_gap,valid50Hzframes=n))
    lineage = dict(checkpoint_sha256=done['checkpoint_sha256'],normalization_sha256=m.sha(npth),
                   train_ids_sha256=EXPECTED['fit_ids'],frozen=True,forward_replay_max_gap=gap,
                   scope='Own FIT14515 selection trunk; not historical full-TRAIN best model',
                   raw_sha256=old_protocol['raw_sha256'],Audio1_source='original1500-frame50Hz mono cache; valid trim after complete-context encoding',
                   Audio1_entries=records,baseline_retrained=False,compact25Hz_cache_not_used=True)
    return logits.detach(),heard,lineage


def load_matchers(args, m, device):
    report = m.read(args.matching/'report.json'); result = {}
    for arm in ('true_pair','wrong_positive','untrained_matcher'):
        if arm == 'untrained_matcher':
            name = 'untrained.pt'; expected = m.read(args.matching/'pre_evaluation_commit.json')['untrained_sha256']
        else:
            train = report['training'][arm]; name = train['checkpoint']; expected = CHECKPOINTS[arm]
            m.require(train['epochs_completed'] == 3 and train['optimizer_steps'] == 192
                      and train['initial_parameter_sha256'] == EXPECTED_INITIAL,'Unequal matcher training')
        path = args.matching/name
        m.require(path.resolve().parent == args.matching.resolve() and m.sha(path) == expected,'Matching checkpoint content changed')
        ck = torch.load(path,map_location='cpu',weights_only=True)
        m.require(ck['protocol_sha256'] == MATCHER_PROTOCOL,'Wrong matching checkpoint protocol')
        if arm != 'untrained_matcher':
            m.require(ck['arm'] == arm and ck['epoch'] == 3,'Wrong final matching epoch')
        else:
            m.require(state_hash(ck['model']) == EXPECTED_INITIAL,'Untrained state not original initialization')
        model = Matcher().to(device); model.load_state_dict(ck['model']); model.requires_grad_(False).eval()
        result[arm] = model
    return result
