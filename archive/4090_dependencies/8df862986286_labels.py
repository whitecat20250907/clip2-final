"""Conservative TRAIN-only human word labels; not the official BEEP scorer."""
from collections import Counter
import math
import re


def expanded_words(text, contractions):
    # Identical to the pinned bank's scoring-word identity contract.
    result = []
    for token in text.split():
        cleaned = re.sub(r'[;!*#,.′’‘_()]', '', token.lower()).strip().replace('’', "'")
        result.extend(contractions.get(cleaned, cleaned).split())
    return result


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stable_alignment(reference, heard):
    """All minimum-edit paths, with ambiguity masked and match-count range.

    Substitution/deletion/insertion cost one, exact equality costs zero. Do not
    choose an alignment because its count matches the released hit count.
    """
    n, m = len(reference), len(heard)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n: cost[i][j] = m - j
            elif j == m: cost[i][j] = n - i
            else:
                cost[i][j] = min(1 + cost[i + 1][j], 1 + cost[i][j + 1],
                                 int(reference[i] != heard[j]) + cost[i + 1][j + 1])
    reached = {(0, 0): (0, 0)}
    possibilities = [set() for _ in reference]
    for i in range(n + 1):
        for j in range(m + 1):
            if (i, j) not in reached: continue
            edges = []
            if i < n and cost[i][j] == 1 + cost[i + 1][j]: edges.append((i + 1, j, 0, i))
            if j < m and cost[i][j] == 1 + cost[i][j + 1]: edges.append((i, j + 1, 0, None))
            if i < n and j < m:
                same = int(reference[i] == heard[j])
                if cost[i][j] == 1 - same + cost[i + 1][j + 1]: edges.append((i + 1, j + 1, same, i))
            lo, hi = reached[i, j]
            for ii, jj, hit, pos in edges:
                old = reached.get((ii, jj), (n + 1, -1))
                reached[ii, jj] = (min(old[0], lo + hit), max(old[1], hi + hit))
                if pos is not None: possibilities[pos].add(hit)
    return ([float(p == {1}) for p in possibilities], [len(p) == 1 for p in possibilities],
            reached[n, m])


def label_row(public, responses, score, contractions, tolerance=1e-5):
    words = public['words']; n = len(words)
    empty = dict(labels=[0.] * n, mask=[False] * n)
    def reject(reason): return dict(**empty, reason=reason)
    if not n or expanded_words(public['prompt'], contractions) != words or public['n_words'] != n:
        raise ValueError('Bank scoring-word identity differs')
    if not isinstance(responses, list) or len(responses) != 1 or not isinstance(responses[0], dict):
        return reject('response_container_or_count')
    row = responses[0]; answer = row.get('response')
    # Never recover a NaN from original_response or reinterpret xxx as silence.
    if not isinstance(answer, str) or answer.strip().lower() == 'xxx':
        return reject('missing_nonfinite_or_marker_response')
    heard = expanded_words(answer, contractions)
    if len(heard) > 200 or n > 100: return reject('bounded_alignment_size')
    lab, keep, count_range = stable_alignment(words, heard)
    if count_range[0] != count_range[1]: return reject('ambiguous_alignment_hit_count')
    hits = row.get('hits'); response_score = row.get('correctness')
    count = count_range[0]
    if not (finite_number(hits) and hits == count and finite_number(response_score)
            and finite_number(score) and 0 <= response_score <= 1 and 0 <= score <= 1
            and abs(response_score - count / n) <= tolerance and abs(score - count / n) <= tolerance):
        return reject('official_hits_or_score_mismatch')
    return dict(labels=lab, mask=keep, reason='accepted')


def scoped_labels(ids, allowed, public, responses, scores, contractions):
    if len(ids) != len(set(ids)) or not set(ids) <= set(allowed):
        raise ValueError('Auxiliary label scope exceeds TRAIN')
    result = {}; counts = Counter()
    for sid in ids:
        # No iteration/dereference of any unselected response value.
        value = label_row(public[sid], responses.get(sid), scores[sid], contractions)
        result[sid] = value; counts[value['reason']] += 1
    summary = dict(rows=len(ids), reasons=dict(counts), rows_with_unambiguous_words=0,
                   unambiguous_words=0, positive_words=0, negative_words=0, ambiguous_words=0)
    for value in result.values():
        summary['rows_with_unambiguous_words'] += int(any(value['mask']))
        summary['ambiguous_words'] += sum(not x for x in value['mask']) if value['reason']=='accepted' else 0
        for y, keep in zip(value['labels'], value['mask']):
            if keep:
                summary['unambiguous_words'] += 1
                summary['positive_words' if y else 'negative_words'] += 1
    return result, summary
