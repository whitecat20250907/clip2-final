"""Read only owned, committed final-L32 Audio2 caches; no extraction fallback."""
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
import numpy as np
import torch
from grounding import VOCAB

DEV_HASH = '6c486a943337877860c67ac871194c84481944106d54e057c15ab556e96149a3'
FIT_HASH = '012dccfa168f824137814af89bfb8168f86bcac3f4e6b1487f8a210c15bc6078'
DICT_HASH = '81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22'
TRAIN_HASH = '4c2a30ff850d55fb1befc2e4c0d9f32002dd2de647a4387338204964534f9b7a'
EXCLUDED_HASH = 'f49c3aadc9030817d93b8ca485617a6d4e179cc45cb98e12d4978bd3a261db2e'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024**2), b''): h.update(part)
    return h.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def lexicon(path):
    require(sha(path) == DICT_HASH, 'Audited dictionary changed')
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split('#', 1)[0].split()
        if len(parts) < 2 or parts[0].startswith(';;;') or '(' in parts[0]: continue
        result.setdefault(parts[0].lower(), [re.sub(r'\d+$', '', p) for p in parts[1:]])
    return result


def word_phones(word, dictionary):
    if word in dictionary: return dictionary[word]
    if word == 'streetlights' and all(p in dictionary for p in ('street', 'lights')):
        return dictionary['street'] + dictionary['lights']
    if word.endswith("in'") or word.endswith('in'):
        value = dictionary.get(word.rstrip("'") + 'g', [])
        if value and value[-1] == 'NG': return value[:-1] + ['N']
    raise ValueError('Unresolved pronunciation: ' + word)


def blocks(text, dictionary, expected):
    text = unicodedata.normalize('NFKC', text).lower().replace('’', "'").replace('‘', "'")
    require(not re.search(r'[^a-z\s\.,!?;:\-—–\'"()…]', text), 'Unspecified lyric normalization')
    words = re.findall(r"[a-z]+(?:'[a-z]+)*'?", text)
    parts = [word_phones(w, dictionary) for w in words]
    flat = [p for part in parts for p in part]
    require(flat == expected and set(flat) <= set(VOCAB), 'Original sung phone query differs')
    return words, parts


def text_arrays(parts):
    ids = [VOCAB[p] for block in parts for p in block]
    starts = [int(i == 0) for block in parts for i in range(len(block))]
    return ids, starts


def reordered(parts):
    original = text_arrays(parts)[0]
    # First changed cyclic word rotation; reject phone-equivalent/homophone order.
    for j in range(1, len(parts)):
        value = parts[j:] + parts[:j]
        if text_arrays(value)[0] != original: return value
    return None


def prepare(args):
    assets = args.assets.resolve(); p = read(assets / 'protocol.json')
    partition = p['partition']
    require(len(partition['fit_ids']) == 14515 and digest(partition['fit_ids']) == FIT_HASH,
            'Must use existing FIT14515 selection population')
    require(len(partition['dev_ids']) == 4059 and digest(partition['dev_ids']) == DEV_HASH,
            'Original4059DEV identity/order changed')
    require(len(partition['train_ids']) == 16204 and len(partition['excluded_train_ids']) == 34,
            'Reviewed retained TRAIN/exclusions changed')
    require(digest(partition['train_ids']) == TRAIN_HASH and digest(partition['excluded_train_ids']) == EXCLUDED_HASH,
            'Reviewed TRAIN/excluded IDs and order changed')
    require(set(partition['fit_ids']).isdisjoint(partition['dev_ids'])
            and set(partition['fit_ids']) <= set(partition['train_ids'])
            and set(partition['train_ids']).isdisjoint(partition['dev_ids']), 'Split overlap')
    require(p['id_hashes'] == {k: digest(v) for k, v in partition.items()}, 'Partition hash changed')
    public = read(assets / 'public.json')
    require(digest(public) == p['public_sha256'], 'Public input mapping changed')
    grouping = Path(p['readiness']) / 'grouping.json'
    require(sha(grouping) == p['grouping_sha256'], 'Recorded exclusion/grouping changed')
    groups = read(grouping)
    recorded = set(groups['excluded_ids']) - set(partition['dev_ids'])
    require(recorded == set(partition['excluded_train_ids']), 'Actual34excluded identities changed')
    group_of = {sid: digest(sorted(g['signals'])) for g in groups['groups'] for sid in g['signals']}
    plan_path = Path(p['source']) / 'results/clip2_full_v1/plan.json'
    require(sha(plan_path) == p['plan_sha256'], 'Original plan changed')
    plan = read(plan_path)
    require(plan['dev_ids'] == partition['dev_ids'], 'Plan DEV order changed')
    plan_rows = {r['signal']: r for r in plan['rows']}
    dictionary_path = args.dictionary
    if not dictionary_path.is_file() and dictionary_path.name == 'cmudict.dict':
        alternative = dictionary_path.parent / 'transfer_lexicon/cmudict.dict'
        if alternative.is_file(): dictionary_path = alternative
    dictionary = lexicon(dictionary_path)
    rows, rejected, seen_pcm, seen_group = [], {}, set(), set()
    ordered = sorted(partition['fit_ids'], key=lambda sid: digest(['grounding-probe-v1', sid]))
    for sid in ordered:
        row = plan_rows[sid]; info = row['files']['audio2']
        require(row['original_prompt'] == public[sid]['original_prompt'], 'Public sung lyric differs')
        if info['pcm_sha256'] in seen_pcm or group_of[sid] in seen_group: continue
        q = plan['queries'][row['query_id']]
        require(q['original_prompt'] == row['original_prompt'], 'Query sung text differs')
        try:
            require(bool(q['tokens']), 'Unresolved source query')
            words, parts = blocks(row['original_prompt'], dictionary, q['tokens'])
            negative = reordered(parts)
            require(negative is not None, 'No changed word-order negative')
            require(0 < len(q['tokens']) <= math.ceil(info['seconds'] / .02), 'Too few frames')
        except ValueError as exc:
            rejected[sid] = str(exc); continue
        seen_pcm.add(info['pcm_sha256']); seen_group.add(group_of[sid])
        rows.append(dict(signal=sid, group=group_of[sid], audio=info, words=words,
                         parts=parts, positive=text_arrays(parts), order_negative=text_arrays(negative)))
        if len(rows) == 32: break
    require(len(rows) == 32, 'Need32 eligible distinct FIT groups for this bounded probe')
    # No selection based on cache presence, labels, severity or model scores.
    for row in rows:
        candidates = [r for r in rows if r['group'] != row['group'] and r['positive'][0] != row['positive'][0]
                      and len(r['positive'][0]) <= math.ceil(row['audio']['seconds'] / .02)]
        require(bool(candidates), 'No nonhomophonic other-lyric negative')
        other = min(candidates, key=lambda r: (abs(len(r['positive'][0]) - len(row['positive'][0])),
                        abs(r['audio']['seconds'] - row['audio']['seconds']), r['signal']))
        row['other_negative'] = other['positive']; row['other_signal'] = other['signal']
    root = args.cache.resolve(); final = read(root / 'protocol.json')
    require(final['schema'] == 'phone_memory_valid_frozen_l32_v1' and final['encoder_frozen']
            and final['trim_after_full_context'] and final['labels_used'] is False, 'Not final frozen L32 cache')
    prefix_path = root.parent / 'phone_memory_prefix_cache_v1/protocol.json'
    prefix = read(prefix_path)
    # Historical cache fingerprint has no allow_nan=False but the values are finite JSON.
    require(digest(prefix) == final['prefix_protocol_sha256'] and prefix['plan_sha256'] == p['plan_sha256'],
            'Final L32 cache/plan lineage differs')
    owner_code = root.parent.parent
    require(sha(owner_code / 'frozen_runtime.py') == final['implementation_sha256'],
            'Recorded final-layer implementation changed')
    for name, expected in prefix['code_sha256'].items():
        require(Path(name).name == name and sha(owner_code / name) == expected, 'Recorded prefix implementation changed')
    require(sha(Path(p['source']) / 'results/clip2_full_v1/feature_protocol.json') == prefix['feature_protocol_sha256'],
            'Original feature extraction protocol changed')
    frames, missing, identities = [], [], []
    for row in rows:
        key = digest(['full-context-prefix30-v1', row['audio']['pcm_sha256'], 0])
        meta_path = root / key[:2] / (key + '.json'); path = root / key[:2] / (key + '.npz')
        if not meta_path.is_file() or not path.is_file():
            missing.append(dict(signal=row['signal'], key=key)); continue
        m = read(meta_path); stat = path.stat()
        require(m['protocol_sha256'] == digest(final) and m['kind'] == 'frames'
                and m['stat'] == [stat.st_size, stat.st_mtime_ns] and sha(path) == m['sha256'],
                'Committed final-L32 content/identity changed')
        with np.load(path, allow_pickle=False) as z:
            require(z.files == ['frames'], 'Unexpected final-cache fields')
            f = z['frames'].copy()
        require(f.dtype == np.float16 and f.shape == (math.ceil(row['audio']['seconds'] / .02), 1280)
                and np.isfinite(f).all(), 'Invalid final-L32 frames')
        frames.append(f.astype(np.float32)); identities.append(dict(signal=row['signal'], path=str(path), **m))
    return rows, frames, dict(schema='grounding_probe_inputs_v1', selected_ids=[r['signal'] for r in rows],
        fit_ids_sha256=FIT_HASH, formal_dev_ids_sha256=DEV_HASH, eligibility_rejected=rejected,
        source_assets_protocol_sha256=sha(assets / 'protocol.json'), plan_sha256=p['plan_sha256'],
        grouping_sha256=p['grouping_sha256'], dictionary_sha256=DICT_HASH, dictionary_path=str(dictionary_path.resolve()),
        final_cache_protocol_sha256=sha(root / 'protocol.json'), cache_entries=identities, missing_cache=missing,
        audio_stream='Audio2 left channel0, complete-context frozen L32, all valid50Hz frames; not dry vocals',
        source_group_or_artist_separation_verified=False, pretrained_weights_rehashed=False,
        fit_scores_read=False, dev_scores_read=False, listener_responses_read=False,
        full_public_plan_parsed=True, full_public_mapping_parsed=True, selected_by_cache_presence=False)


def collate(frames, texts, device):
    b = len(frames); t = max(map(len, frames)); n = max(len(x[0]) for x in texts)
    x = np.zeros((b, t, 1280), np.float32); ids = np.zeros((b, n), np.int64); starts = ids.copy()
    for i, (f, (ph, bounds)) in enumerate(zip(frames, texts)):
        require(len(f) >= len(ph), 'Negative phone sequence longer than valid audio')
        x[i, :len(f)] = f; ids[i, :len(ph)] = ph; starts[i, :len(ph)] = bounds
    make = lambda a: torch.from_numpy(a).to(device)
    return (make(x), torch.tensor([len(f) for f in frames], device=device), make(ids), make(starts),
            torch.tensor([len(z[0]) for z in texts], device=device))
