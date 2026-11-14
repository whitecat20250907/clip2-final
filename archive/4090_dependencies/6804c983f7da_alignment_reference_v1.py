"""Reference-word alignment and audited human-response supervision.

Use the dataset's already-normalized ``prompt`` and ``response`` fields.
Normalization here only case-folds and splits whitespace: it does not guess
contractions, remove punctuation, or reinterpret special response markers.

A minimum Levenshtein alignment is not always unique. We return one
deterministic alignment and audit *all* optimal paths for ambiguous reference
labels/mappings. Human supervision masks ambiguous labels, but sentence
targets remain the caller's original metadata targets.
"""

import math


def _words(text):
    if not isinstance(text, str):
        raise TypeError("Expected text, including an explicit empty string.")
    return text.casefold().split()


def align_words(reference: str, hypothesis: str) -> dict:
    """Align words with unit-cost substitution, deletion and insertion.

    ``labels[i]`` is 1 only for an exact match of reference word i.
    ``hyp_indices[i]`` is its hypothesis index, or None for a deletion.
    Backtrace ties prefer a diagonal step, then deletion, then insertion.
    ``label_mask`` excludes words whose correctness differs across equally
    optimal paths. ``mapping_mask`` also excludes differing token mappings.

    Complexity is O(number_of_reference_words * number_of_hypothesis_words).
    """
    ref, hyp = _words(reference), _words(hypothesis)
    n, m = len(ref), len(hyp)
    forward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        forward[i][0] = i
    for j in range(m + 1):
        forward[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            forward[i][j] = min(
                forward[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                forward[i - 1][j] + 1,
                forward[i][j - 1] + 1,
            )

    labels, hyp_indices = [0] * n, [None] * n
    substitutions = deletions = insertions = 0
    operations = []
    i, j = n, m
    while i or j:
        if i and j and forward[i][j] == (
            forward[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
        ):
            match = ref[i - 1] == hyp[j - 1]
            labels[i - 1] = int(match)
            hyp_indices[i - 1] = j - 1
            substitutions += int(not match)
            operations.append(("match" if match else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and forward[i][j] == forward[i - 1][j] + 1:
            deletions += 1
            operations.append(("del", i - 1, None))
            i -= 1
        else:
            insertions += 1
            operations.append(("ins", None, j - 1))
            j -= 1

    backward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        backward[i][m] = n - i
    for j in range(m + 1):
        backward[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            backward[i][j] = min(
                backward[i + 1][j + 1] + (ref[i] != hyp[j]),
                backward[i + 1][j] + 1,
                backward[i][j + 1] + 1,
            )

    distance = forward[n][m]
    choices = [set() for _ in range(n)]
    for i in range(n):
        for j in range(m + 1):
            if forward[i][j] + 1 + backward[i + 1][j] == distance:
                choices[i].add((0, None))
            if j < m:
                match = ref[i] == hyp[j]
                if forward[i][j] + (not match) + backward[i + 1][j + 1] == distance:
                    choices[i].add((int(match), j))

    label_mask = [len({label for label, _ in options}) == 1 for options in choices]
    mapping_mask = [len(options) == 1 for options in choices]
    return {
        "labels": labels,
        "hyp_indices": hyp_indices,
        "hits": sum(labels),
        "s": substitutions,
        "d": deletions,
        "i": insertions,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "distance": distance,
        "reference_words": ref,
        "hypothesis_words": hyp,
        "operations": list(reversed(operations)),
        "label_mask": label_mask,
        "mapping_mask": mapping_mask,
        "label_ambiguous_indices": [i for i, valid in enumerate(label_mask) if not valid],
        "mapping_ambiguous_indices": [i for i, valid in enumerate(mapping_mask) if not valid],
    }


def _integer_count(value, minimum=0):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if not math.isfinite(number) or number < minimum or not number.is_integer():
        return None
    return int(number)


_UNSET = object()


def _finite_score(value):
    if value is _UNSET or value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def validate_human_response(prompt, response, n_words, hits, *,
                            response_correctness=_UNSET,
                            metadata_correctness=_UNSET) -> dict:
    """Produce labels/mask without changing any sentence-level target.

    An observed boundary count (hits=0 or hits=n_words) identifies every word
    label without a transcript, but ONLY if both official correctness fields
    are supplied, finite, and agree with hits/n_words. Missing hits never
    defaults to zero. Intermediate counts cannot locate missing word labels.

    Inconsistent official word/hit counts mask *all* word supervision for the
    sample. With consistent counts, only ambiguous reference labels are
    masked. ``valid`` means metadata counts agree, not that every word is
    usable. ``label_source`` and ``recovered_words`` distinguish transcript
    alignment from count-derived supervision. ``alignment`` always retains
    the original transcript diagnostic, even when count-derived labels differ.

    For backwards compatibility, a four-argument call performs only the
    original transcript/count audit and NEVER enables count-based recovery.
    Production bundle/audit callers must provide both correctness keywords.
    """
    if not isinstance(prompt, str):
        return {"labels": [], "mask": [], "valid": False,
                "reason": "invalid_prompt", "alignment": None,
                "label_source": "unavailable", "recovered_words": 0,
                "recovered": False, "raw_alignment_reason": "invalid_prompt",
                "response_missing": not isinstance(response, str)}
    response_present = isinstance(response, str)
    alignment = align_words(prompt, response if response_present else "")
    labels = alignment["labels"]
    expected_n = _integer_count(n_words, minimum=1)
    expected_hits = _integer_count(hits)
    official_reasons, text_reasons = [], []
    if expected_n is None:
        official_reasons.append("invalid_n_words")
    elif expected_n != len(labels):
        official_reasons.append("n_words_mismatch")
    if expected_hits is None:
        official_reasons.append("invalid_hits")
    elif expected_n is not None and expected_hits > expected_n:
        official_reasons.append("hits_exceed_n_words")

    scores_supplied = response_correctness is not _UNSET or metadata_correctness is not _UNSET
    count_score = (expected_hits / expected_n) if expected_n is not None and expected_hits is not None else None
    if scores_supplied:
        response_score = _finite_score(response_correctness)
        metadata_score = _finite_score(metadata_correctness)
        if response_score is None:
            official_reasons.append("invalid_response_correctness")
        elif count_score is not None and not math.isclose(response_score, count_score, rel_tol=0, abs_tol=1e-7):
            official_reasons.append("response_correctness_count_mismatch")
        if metadata_score is None:
            official_reasons.append("invalid_metadata_correctness")
        elif count_score is not None and not math.isclose(metadata_score, count_score, rel_tol=0, abs_tol=1e-7):
            official_reasons.append("metadata_correctness_count_mismatch")
        if response_score is not None and metadata_score is not None and not math.isclose(
            response_score, metadata_score, rel_tol=0, abs_tol=1e-7
        ):
            official_reasons.append("response_metadata_correctness_mismatch")

    if not response_present:
        is_nan = isinstance(response, float) and math.isnan(response)
        text_reasons.append("missing_response" if response is None or is_nan else "invalid_response")
    if expected_hits is not None and expected_hits != alignment["hits"]:
        text_reasons.append("hits_mismatch")
    raw_reason = ";".join(text_reasons) if text_reasons else (
        "ambiguous_word_labels" if not all(alignment["label_mask"]) else "ok")

    if scores_supplied and not official_reasons and expected_hits in (0, expected_n):
        source = "count_zero" if expected_hits == 0 else "count_full"
        recovered_words = len(labels) if text_reasons else sum(not value for value in alignment["label_mask"])
        return {"labels": [int(expected_hits == expected_n)] * len(labels),
                "mask": [True] * len(labels), "valid": True, "reason": source,
                "alignment": alignment, "label_source": source,
                "recovered_words": recovered_words, "recovered": recovered_words > 0,
                "raw_alignment_reason": raw_reason, "response_missing": not response_present}

    reasons = official_reasons + text_reasons
    valid = not reasons
    mask = list(alignment["label_mask"]) if valid else [False] * len(labels)
    if valid and not all(mask):
        reason = "ambiguous_word_labels"
    else:
        reason = ";".join(reasons) if reasons else "ok"
    return {"labels": labels, "mask": mask, "valid": valid,
            "reason": reason, "alignment": alignment,
            "label_source": "transcript_alignment" if valid else "unavailable",
            "recovered_words": 0, "recovered": False,
            "raw_alignment_reason": raw_reason, "response_missing": not response_present}
