"""Reconstruct CLIP1 response alternatives from the organizers' published tables.

This is a technical replay, never a search over released hit counts.  Original
human responses are expanded before choosing the maximum scoring alternative,
as in the official tutorial.  No CLIP1 VALID/TEST/EVAL labels are opened.
"""
from pathlib import Path
import os, sys, re, json, collections, itertools, hashlib, datetime, time

J = Path(__file__).parent
sys.path[:0] = [str(J / 'v10_final15h_site'), str(J / 'v10_z13_scoring_site'), str(J)]
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import inflect
from v10_Y1_replay_ties_20261022 import replay

V = J.parent / 'CLIP2-new/algorithms/03_my_improvements/experiments/v10_20261016'
D = V / 'third_round_20261023'
OUT = D / 'Z13_repair'
ASSETS = OUT / 'official_scoring_sources'
TOKEN = re.compile(r"[a-z]+(?:['\u2019][a-z]+)*(?:['\u2019])?|['\u2019][a-z]+|[^\w\s]", re.I)
NUM = inflect.engine()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, multiple=True):
    out = collections.defaultdict(list)
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        parts = [p.strip().lower() for p in line.strip().split(',', 1)]
        if not parts or not parts[0]:
            continue
        key, value = parts if len(parts) == 2 else (parts[0], '')
        out[key].append(value)
    return dict(out) if multiple else {k: v[-1] for k, v in out.items()}


SPELL = table(ASSETS / 'spelling_corrections.csv', multiple=False)
ALT = table(ASSETS / 'alternative_words.csv')
CONTRACTIONS = table(ASSETS / 'contractions.csv')


def variants(text, tab):
    # Mirrors AlternativeWords.make_sentence_forms in the published tutorial.
    tokens = TOKEN.findall(text.lower())
    choices = [tab.get(w, []) + [w] for w in tokens]
    return sorted(set(' '.join(v).strip() for v in itertools.product(*choices)))


def clean(text):
    def spelling(s):
        return re.sub(r'[^\s]+', lambda m: SPELL.get(m[0].lower(), m[0]), s)

    text = spelling(text)
    text = re.sub(r'\d+\.\d+', lambda m: NUM.number_to_words(m[0].split('.')[0], andword='').replace('-', ' ') + ' point ' + ' '.join(NUM.number_to_words(d) for d in m[0].split('.')[1]), text)
    text = re.sub(r'\b\d+\b', lambda m: NUM.number_to_words(m[0], andword='').replace('-', ' '), text).lower()
    text = re.sub(r'[;!*#,.′’‘_()]', '', text)
    text = spelling(text)
    return sorted(set(' '.join(c.split()) for alt in variants(text, ALT) for c in variants(alt, CONTRACTIONS)))


def main():
    start = time.monotonic()
    rows = json.loads((D / 'assets/clip1_train_metadata.json').read_bytes())
    lex = json.loads((V / 'final15h_20261022/Y1/beep_dictionary.json').read_bytes())
    stats = collections.Counter()
    labels, bad = {}, []
    max_variants = 0
    for i, r in enumerate(rows):
        ref = r['prompt'].lower().split()  # Preserve the released '?' word slots.
        hyps = clean(r['original_response'])
        max_variants = max(max_variants, len(hyps))
        scored = []
        for h in hyps:
            yy, mapping = replay(ref, h.split(), lex, 'rapid_all')
            scored.append((sum(yy), h, yy, mapping))
        # Official maximum-hit rule.  Ties use fixed lexical ordering, never
        # released words_correct or a per-row choice of alignment rules.
        best = min(scored, key=lambda t: (-t[0], t[1]))
        hits, hyp, yy, mapping = best
        ok = hits == int(r['words_correct']) and len(ref) == int(r['n_words'])
        stats['rows'] += 1
        stats['matched'] += ok
        stats['blank_answers'] += not r['response'].strip()
        stats['blank_answers_retained'] += not r['response'].strip()
        stats['too_many'] += hits > int(r['words_correct'])
        stats['too_few'] += hits < int(r['words_correct'])
        stats['length_mismatch'] += len(ref) != int(r['n_words'])
        labels[r['signal']] = dict(reference=ref, response=hyp, labels=yy, mapping=mapping, hits=hits, released_hits=r['words_correct'], hits_match=ok, hearing_loss=r['hearing_loss'], content=str(r['fma'])+'|'+' '.join(ref))
        if not ok:
            bad.append({k: r[k] for k in ['signal', 'original_prompt', 'prompt', 'original_response', 'response', 'n_words', 'words_correct']} | dict(actual=hits, best_alternative=hyp, alternative_count=len(hyps)))
        if i % 1000 == 0:
            write(OUT / 'replay_progress.json', dict(rows=i, total=len(rows), seconds=time.monotonic()-start))
    report = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **stats, fraction=stats['matched']/stats['rows'], threshold=.99, maximum_response_variants=max_variants, seconds=time.monotonic()-start, source_provenance=json.loads((ASSETS/'provenance.json').read_bytes()), CLIP1_TRAIN_only=True, VALID_or_EVAL_labels=False, no_per_row_hit_count_fitting=True, reference_pronunciation='BEEP variant intersection: released random reference pronunciation unavailable; count replay does not establish unique alignment', official_tutorial='https://cadenzachallenge.org/cadenza_tutorials/clip1/scoring/scoring.html', scorer_SHA=sha(Path(__file__)), released_targets_never_modified=True)
    write(OUT / 'replay_report.json', report)
    write(OUT / 'remaining_mismatches.json', bad)
    write(OUT / 'replayed_word_labels.json', labels)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print('MISMATCH_EXAMPLES', json.dumps(bad[:12], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
