"""D6e + level + generic lyric controls + reference/ASR phone errors.

The registered arms are:
  overall:   per-ear overall phone error rates;
  stop:      overall plus per-ear voiceless-stop error burden;
  fricative: overall plus per-ear fricative error burden;
  sonorant:  overall plus per-ear sonorant error burden;
  stop_fricative, stop_sonorant, fricative_sonorant: two-class ablations;
  consonant: the same rates plus per-ear stop/fricative/sonorant error burden.
  class_rate: replace the six burdens with within-class conditional error rates.
  consonant_lyrics: keep burdens but replace generic text controls with lyric phonetic demand.
  strict_raw: exact-token phone classes with the original raw left/right representation.
  strict_be_gap: strict classes represented as better-ear minimum plus absolute ear gap.
  strict_smoothed: strict per-class conditional rates shrunk toward each ear's overall rate.
  strict_factorized: replace generic text controls with phone count and strict class demand.
  strict_burden_demand: strict raw burdens plus phone count and strict class demand.
  manner3_burden_demand: full stop/affricate, fricative, and sonorant burdens plus demand.
  strict_deletion_emphasis_demand: strict deletion burden plus half-weight substitutions.
  position_burden_demand: word-onset, medial, and coda consonant burdens plus demand.
  strict_burden_demand_branch: current best features through a parameter-matched phone branch.
  strict_burden_level_interaction: strict burdens modulated by matched-band level distortion.
  top5_phone_burden_demand: five frequent phones plus an exhaustive other-phone group.
  strict_dual_burden_rate: strict demand with both burden and smoothed conditional rates.
  strict_joint_error_demand: shared versus complementary reference-phone errors across ears.

The frozen Whisper vector, existing D6e side features, 306-D regressor, loss,
data split and checkpoint selection are inherited from the paired reference
experiment. This is text alignment between reference lyrics and ASR hypotheses;
it does not imply reliable time alignment to sung audio or human phone labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
BASE_CONFIG = EXPERIMENTS / "d6e_paired_reference_v1/D6e_paired_config.yaml"
ABLATION_HELPER = EXPERIMENTS / "d6e_paired_feature_ablation_v1/run_ablation.py"
PHONETIC_DEMAND_CACHE = (
    EXPERIMENTS / "d6e_ref_lyric_consonants_v1/reference_lyric_phonetic_demand.json"
)
CANDIDATE_SCALAR_CACHE = EXPERIMENTS / "shared_feature_cache/candidate_max_overall.jsonl"
WORD_CACHE = HERE / "word_ipa_cache.json"
FEATURE_CACHE = HERE / "phone_error_features.jsonl"
FEATURE_MANIFEST = HERE / "phone_error_features_manifest.json"
RATE_FEATURE_CACHE = HERE / "phone_class_rate_features.jsonl"
RATE_FEATURE_MANIFEST = HERE / "phone_class_rate_features_manifest.json"
STRICT_FEATURE_CACHE = HERE / "phone_strict_features.jsonl"
STRICT_FEATURE_MANIFEST = HERE / "phone_strict_features_manifest.json"
STRICT_FACTORIZED_CACHE = HERE / "phone_strict_factorized_features.jsonl"
STRICT_FACTORIZED_MANIFEST = HERE / "phone_strict_factorized_features_manifest.json"
MANNER3_CACHE = HERE / "phone_manner3_burden_demand_features.jsonl"
MANNER3_MANIFEST = HERE / "phone_manner3_burden_demand_features_manifest.json"
ERROR_TYPE_CACHE = HERE / "phone_strict_error_type_features.jsonl"
ERROR_TYPE_MANIFEST = HERE / "phone_strict_error_type_features_manifest.json"
POSITION_CACHE = HERE / "phone_position_burden_demand_features.jsonl"
POSITION_MANIFEST = HERE / "phone_position_burden_demand_features_manifest.json"
TOP5_CACHE = HERE / "phone_top5_burden_demand_features.jsonl"
TOP5_MANIFEST = HERE / "phone_top5_burden_demand_features_manifest.json"
STRICT_JOINT_CACHE = HERE / "phone_strict_joint_error_demand_features.jsonl"
STRICT_JOINT_MANIFEST = HERE / "phone_strict_joint_error_demand_features_manifest.json"

LEVEL_INDICES = (0, 1, 2, 3, 10, 11, 12, 13)
EXTRA_INDICES = (4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19)
TEXT_FEATURE_NAMES = (
    "lyric_word_count",
    "lyric_mean_letters_per_word",
    "lyric_short_word_share",
    "lyric_unique_word_share",
)
PHONETIC_DEMAND_NAMES = (
    "word_share_voiceless_stops_ptk",
    "word_share_fricatives",
    "word_share_voiced_sonorants",
    "word_share_consonant_coda",
)
PHONE_FEATURE_NAMES = (
    "left_phone_error_rate",
    "right_phone_error_rate",
    "left_voiceless_stop_error_burden",
    "left_fricative_error_burden",
    "left_sonorant_error_burden",
    "right_voiceless_stop_error_burden",
    "right_fricative_error_burden",
    "right_sonorant_error_burden",
)
RATE_FEATURE_NAMES = (
    "left_phone_error_rate",
    "right_phone_error_rate",
    "left_voiceless_stop_conditional_error_rate",
    "left_fricative_conditional_error_rate",
    "left_sonorant_conditional_error_rate",
    "right_voiceless_stop_conditional_error_rate",
    "right_fricative_conditional_error_rate",
    "right_sonorant_conditional_error_rate",
)
STRICT_FEATURE_NAMES = (
    "left_phone_error_rate",
    "right_phone_error_rate",
    "left_strict_voiceless_stop_error_burden",
    "left_strict_fricative_affricate_error_burden",
    "left_strict_sonorant_consonant_error_burden",
    "right_strict_voiceless_stop_error_burden",
    "right_strict_fricative_affricate_error_burden",
    "right_strict_sonorant_consonant_error_burden",
)
STRICT_BE_GAP_FEATURE_NAMES = (
    "strict_overall_better_ear_error_rate",
    "strict_overall_absolute_ear_gap",
    "strict_voiceless_stop_better_ear_error_burden",
    "strict_voiceless_stop_absolute_ear_gap",
    "strict_fricative_affricate_better_ear_error_burden",
    "strict_fricative_affricate_absolute_ear_gap",
    "strict_sonorant_consonant_better_ear_error_burden",
    "strict_sonorant_consonant_absolute_ear_gap",
)
STRICT_FACTORIZED_FEATURE_NAMES = (
    "left_phone_error_rate",
    "right_phone_error_rate",
    "reference_log1p_phone_count",
    "reference_strict_voiceless_stop_share",
    "reference_strict_fricative_affricate_share",
    "reference_strict_sonorant_consonant_share",
    "left_strict_voiceless_stop_smoothed_error_rate",
    "left_strict_fricative_affricate_smoothed_error_rate",
    "left_strict_sonorant_consonant_smoothed_error_rate",
    "right_strict_voiceless_stop_smoothed_error_rate",
    "right_strict_fricative_affricate_smoothed_error_rate",
    "right_strict_sonorant_consonant_smoothed_error_rate",
)
STRICT_BURDEN_DEMAND_FEATURE_NAMES = (
    "reference_log1p_phone_count",
    "reference_strict_voiceless_stop_share",
    "reference_strict_fricative_affricate_share",
    "reference_strict_sonorant_consonant_share",
    *STRICT_FEATURE_NAMES,
)
MANNER3_FEATURE_NAMES = (
    "reference_log1p_phone_count",
    "reference_stop_affricate_share",
    "reference_fricative_share",
    "reference_sonorant_consonant_share",
    "left_phone_error_rate",
    "right_phone_error_rate",
    "left_stop_affricate_error_burden",
    "left_fricative_error_burden",
    "left_sonorant_consonant_error_burden",
    "right_stop_affricate_error_burden",
    "right_fricative_error_burden",
    "right_sonorant_consonant_error_burden",
)
ERROR_TYPE_CACHE_FEATURE_NAMES = (
    "reference_log1p_phone_count",
    "reference_strict_voiceless_stop_share",
    "reference_strict_fricative_affricate_share",
    "reference_strict_sonorant_consonant_share",
    "left_phone_error_rate",
    "right_phone_error_rate",
    "left_deletion_rate",
    "left_substitution_rate",
    "left_insertion_rate",
    "right_deletion_rate",
    "right_substitution_rate",
    "right_insertion_rate",
    "left_strict_voiceless_stop_deletion_burden",
    "left_strict_fricative_affricate_deletion_burden",
    "left_strict_sonorant_consonant_deletion_burden",
    "left_strict_voiceless_stop_substitution_burden",
    "left_strict_fricative_affricate_substitution_burden",
    "left_strict_sonorant_consonant_substitution_burden",
    "right_strict_voiceless_stop_deletion_burden",
    "right_strict_fricative_affricate_deletion_burden",
    "right_strict_sonorant_consonant_deletion_burden",
    "right_strict_voiceless_stop_substitution_burden",
    "right_strict_fricative_affricate_substitution_burden",
    "right_strict_sonorant_consonant_substitution_burden",
)
DELETION_EMPHASIS_FEATURE_NAMES = STRICT_BURDEN_DEMAND_FEATURE_NAMES
POSITION_FEATURE_NAMES = (
    "reference_log1p_phone_count",
    "reference_word_onset_consonant_share",
    "reference_word_medial_consonant_share",
    "reference_word_coda_consonant_share",
    "left_phone_error_rate",
    "right_phone_error_rate",
    "left_word_onset_consonant_error_burden",
    "left_word_medial_consonant_error_burden",
    "left_word_coda_consonant_error_burden",
    "right_word_onset_consonant_error_burden",
    "right_word_medial_consonant_error_burden",
    "right_word_coda_consonant_error_burden",
)
TOP5_PHONES = ("t", "n", "ɪ", "s", "l")
TOP5_GROUP_NAMES = ("t", "n", "small_cap_i", "s", "l", "other")
TOP5_FEATURE_NAMES = (
    "left_phone_error_rate",
    "right_phone_error_rate",
    *(f"reference_phone_share_{name}" for name in TOP5_GROUP_NAMES),
    *(f"left_phone_error_burden_{name}" for name in TOP5_GROUP_NAMES),
    *(f"right_phone_error_burden_{name}" for name in TOP5_GROUP_NAMES),
)
STRICT_DUAL_FEATURE_NAMES = (
    "reference_log1p_phone_count",
    "reference_strict_voiceless_stop_share",
    "reference_strict_fricative_affricate_share",
    "reference_strict_sonorant_consonant_share",
    "left_phone_error_rate",
    "right_phone_error_rate",
    "left_strict_voiceless_stop_error_burden",
    "left_strict_fricative_affricate_error_burden",
    "left_strict_sonorant_consonant_error_burden",
    "right_strict_voiceless_stop_error_burden",
    "right_strict_fricative_affricate_error_burden",
    "right_strict_sonorant_consonant_error_burden",
    "left_strict_voiceless_stop_smoothed_error_rate",
    "left_strict_fricative_affricate_smoothed_error_rate",
    "left_strict_sonorant_consonant_smoothed_error_rate",
    "right_strict_voiceless_stop_smoothed_error_rate",
    "right_strict_fricative_affricate_smoothed_error_rate",
    "right_strict_sonorant_consonant_smoothed_error_rate",
)
STRICT_JOINT_FEATURE_NAMES = (
    "reference_log1p_phone_count",
    "reference_strict_voiceless_stop_share",
    "reference_strict_fricative_affricate_share",
    "reference_strict_sonorant_consonant_share",
    "left_phone_error_rate",
    "right_phone_error_rate",
    "both_ears_voiceless_stop_reference_error_burden",
    "both_ears_fricative_affricate_reference_error_burden",
    "both_ears_sonorant_consonant_reference_error_burden",
    "one_ear_only_voiceless_stop_reference_error_burden",
    "one_ear_only_fricative_affricate_reference_error_burden",
    "one_ear_only_sonorant_consonant_reference_error_burden",
)
MODE_PHONE_INDICES = {
    "overall": (0, 1),
    "stop": (0, 1, 2, 5),
    "fricative": (0, 1, 3, 6),
    "sonorant": (0, 1, 4, 7),
    "stop_fricative": (0, 1, 2, 3, 5, 6),
    "stop_sonorant": (0, 1, 2, 4, 5, 7),
    "fricative_sonorant": (0, 1, 3, 4, 6, 7),
    "consonant": tuple(range(8)),
    "class_rate": tuple(range(8)),
    "consonant_lyrics": tuple(range(8)),
    "strict_raw": tuple(range(8)),
    "strict_be_gap": tuple(range(8)),
    "strict_smoothed": (0, 1, 6, 7, 8, 9, 10, 11),
    "strict_factorized": tuple(range(12)),
    "strict_burden_demand": tuple(range(12)),
    "manner3_burden_demand": tuple(range(12)),
    "strict_deletion_emphasis_demand": tuple(range(12)),
    "position_burden_demand": tuple(range(12)),
    "strict_burden_demand_branch": tuple(range(12)),
    "strict_burden_level_interaction": tuple(range(12)),
    "top5_phone_burden_demand": tuple(range(20)),
    "strict_dual_burden_rate": tuple(range(18)),
    "strict_joint_error_demand": tuple(range(12)),
}
STOP = frozenset("ptk")
FRICATIVE = frozenset("fvθðszʃʒhɦçx")
SONORANT = frozenset("lmnŋɲɹrjɥwɾ")
STRICT_STOP = frozenset(("p", "t", "k"))
STRICT_FRICATIVE_AFFRICATE = frozenset(
    ("f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "x", "tʃ", "dʒ")
)
STRICT_SONORANT = frozenset(("j", "l", "m", "n", "w", "ŋ", "ɹ", "ɾ"))
MANNER3_STOP_AFFRICATE = frozenset(
    ("p", "b", "t", "d", "k", "ɡ", "ʔ", "tʃ", "dʒ")
)
MANNER3_FRICATIVE = frozenset(("f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "x"))
MANNER3_SONORANT = STRICT_SONORANT
EXACT_CONSONANTS = MANNER3_STOP_AFFRICATE | MANNER3_FRICATIVE | MANNER3_SONORANT
NORMALIZE_DROP = frozenset("ˈˌːˑ͡")


def load_ablation_helper():
    spec = importlib.util.spec_from_file_location("phone_error_ablation_helpers", ABLATION_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ABLATION_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_phone_branch_regressor(trainer):
    """Create a trainer-compatible regressor with a separate 12->8 phone encoder."""
    nn = trainer.nn
    torch = trainer.torch
    paired_phone_slots = tuple(EXTRA_INDICES[:12])

    class PhoneBranchRegressor(nn.Module):
        def __init__(self, wc, side_dim_v3, side_dim_v2, model_cfg, extra_wer_dim=0):
            super().__init__()
            d = float(model_cfg["dropout"])
            self.extra_wer_dim = int(extra_wer_dim)
            paired_dim = int(model_cfg["phone_branch_paired_dim"])
            prefix_dim = self.extra_wer_dim - paired_dim
            if paired_dim != 20 or prefix_dim != 5:
                raise ValueError(
                    "Phone branch expects 4 extra WER + 1 candidate + 20 paired features"
                )
            phone_indices = tuple(prefix_dim + index for index in paired_phone_slots)
            core_indices = tuple(
                index for index in range(self.extra_wer_dim)
                if index not in phone_indices
            )
            if len(phone_indices) != 12 or len(core_indices) != 13:
                raise RuntimeError("Unexpected phone/core feature partition")
            self.register_buffer(
                "phone_indices", torch.tensor(phone_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "core_indices", torch.tensor(core_indices, dtype=torch.long),
                persistent=False,
            )
            self.whisper_projection = trainer.Projection(
                wc["pooled_dim"], wc["projection_dim"], d
            )
            phone_dim = int(model_cfg["phone_branch_output_dim"])
            self.phone_encoder = nn.Sequential(
                nn.Linear(12, phone_dim),
                nn.GELU(),
                nn.LayerNorm(phone_dim),
                nn.Dropout(d),
            )
            fused = (
                int(wc["projection_dim"])
                + int(side_dim_v3)
                + int(side_dim_v2)
                + 1
                + len(core_indices)
                + phone_dim
            )
            expected = int(model_cfg["phone_branch_internal_fused_dim"])
            if fused != expected:
                raise ValueError(f"Phone branch internal fused dim must be {expected}, got {fused}")
            h = int(model_cfg["regression_hidden_dim"])
            self.regressor = nn.Sequential(
                nn.Linear(fused, h),
                nn.GELU(),
                nn.Dropout(d),
                nn.Linear(h, 1),
                nn.Sigmoid(),
            )

        def forward(self, w, c_v3, c_v2, z, extra_wer=None):
            if extra_wer is None or extra_wer.shape[-1] != self.extra_wer_dim:
                raise ValueError("Phone branch received an invalid extra feature tensor")
            w = self.whisper_projection(w)
            phone = self.phone_encoder(extra_wer.index_select(-1, self.phone_indices))
            core = extra_wer.index_select(-1, self.core_indices)
            return self.regressor(
                torch.cat((w, c_v3, c_v2, z, core, phone), dim=-1)
            ).squeeze(-1)

    return PhoneBranchRegressor


def load_candidate_scalar_cache(required_ids, field="candidate_max_overall"):
    if field != "candidate_max_overall":
        raise ValueError(f"Unsupported cached candidate field: {field}")
    if not CANDIDATE_SCALAR_CACHE.is_file():
        raise FileNotFoundError(
            f"Missing aggregate candidate cache: {CANDIDATE_SCALAR_CACHE}. "
            "Run shared_feature_cache/build_candidate_scalar_cache.py first."
        )
    required = set(map(str, required_ids))
    table = {}
    with CANDIDATE_SCALAR_CACHE.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            sid = str(item["signal"])
            if sid not in required:
                continue
            value = float(item[field])
            if sid in table or not np.isfinite(value):
                raise ValueError(
                    f"Invalid aggregate candidate cache row {line_no}: {CANDIDATE_SCALAR_CACHE}"
                )
            table[sid] = np.float32(value)
    missing = sorted(required - set(table))
    if missing:
        raise RuntimeError(
            f"Aggregate candidate cache missing {len(missing)} required IDs: {missing[:10]}"
        )
    print(
        f"Reusing aggregate candidate cache: {len(table):,} samples | "
        f"{CANDIDATE_SCALAR_CACHE}",
        flush=True,
    )
    return table


def normalized_words(text):
    """Match the base trainer's punctuation-free word-WER convention."""
    import string

    clean = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return tuple(re.findall(r"[a-z0-9]+", clean))


def generic_text_features(prompt):
    # Exactly match d6e_ref_lyric_consonants_v1 --mode text_control.
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", prompt.lower())
    if not words:
        raise ValueError(f"Reference lyric has no English words: {prompt!r}")
    lengths = [sum(ch.isalpha() for ch in word) for word in words]
    return np.asarray(
        [
            len(words),
            float(np.mean(lengths)),
            sum(length <= 3 for length in lengths) / len(words),
            len(set(words)) / len(words),
        ],
        dtype=np.float32,
    )


def load_phonetic_demand(prompts):
    if not PHONETIC_DEMAND_CACHE.is_file():
        raise FileNotFoundError(f"Missing existing phonetic-demand cache: {PHONETIC_DEMAND_CACHE}")
    obj = json.loads(PHONETIC_DEMAND_CACHE.read_text(encoding="utf-8"))
    if obj.get("version") != 1 or obj.get("feature_names") != list(PHONETIC_DEMAND_NAMES):
        raise ValueError("Phonetic-demand cache schema mismatch")
    raw = obj.get("features")
    if not isinstance(raw, dict) or not set(prompts.values()).issubset(raw):
        raise ValueError("Phonetic-demand cache does not cover all reference lyrics")
    table = {}
    for sid, prompt in prompts.items():
        vec = np.asarray(raw[prompt], dtype=np.float32)
        if vec.shape != (4,) or not np.isfinite(vec).all() or not ((0 <= vec) & (vec <= 1)).all():
            raise ValueError(f"Invalid phonetic-demand vector: {sid}")
        table[sid] = vec
    print(f"Reusing reference-lyric phonetic demand: {len(table):,} samples", flush=True)
    return table


def normalize_phone(phone):
    return "".join(
        ch for ch in phone.strip()
        if ch not in NORMALIZE_DROP and not unicodedata.combining(ch)
    )


def phonemize_word(word):
    proc = subprocess.run(
        ["espeak-ng", "-q", "-v", "en-us", "--ipa", "--sep=_", word],
        check=True, capture_output=True, text=True, timeout=15,
    )
    phones = tuple(
        clean for piece in proc.stdout.replace("\n", "_").split("_")
        if (clean := normalize_phone(piece))
    )
    if not phones:
        raise ValueError(f"No IPA phones for {word!r}: {proc.stdout!r}")
    return phones


def phone_class(phone):
    if any(ch in STOP for ch in phone):
        return 0
    if any(ch in FRICATIVE for ch in phone):
        return 1
    if any(ch in SONORANT for ch in phone):
        return 2
    return None


def strict_phone_class(phone):
    """Return an exact-token class; compound/rhotic vowels do not match by substring."""
    if phone in STRICT_STOP:
        return 0
    if phone in STRICT_FRICATIVE_AFFRICATE:
        return 1
    if phone in STRICT_SONORANT:
        return 2
    return None


def manner3_phone_class(phone):
    if phone in MANNER3_STOP_AFFRICATE:
        return 0
    if phone in MANNER3_FRICATIVE:
        return 1
    if phone in MANNER3_SONORANT:
        return 2
    return None


def top5_phone_class(phone):
    try:
        return TOP5_PHONES.index(phone)
    except ValueError:
        return 5


def strict_be_gap_vector(raw):
    """Convert [L/R overall, L classes, R classes] to ear-symmetric features."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (8,):
        raise ValueError(f"Expected 8 strict raw features, got {raw.shape}")
    left_overall, right_overall = float(raw[0]), float(raw[1])
    left_classes, right_classes = raw[2:5], raw[5:8]
    values = [min(left_overall, right_overall), abs(left_overall - right_overall)]
    for left, right in zip(left_classes, right_classes):
        values.extend((min(float(left), float(right)), abs(float(left) - float(right))))
    return np.asarray(values, dtype=np.float32)


def level_interaction_vector(phone, paired):
    """Modulate strict class burdens by the matching ear's band-level distortion."""
    phone = np.asarray(phone, dtype=np.float32)
    paired = np.asarray(paired, dtype=np.float32)
    if phone.shape != (12,) or paired.shape != (20,):
        raise ValueError(f"Expected 12-D phone and 20-D paired features, got {phone.shape}/{paired.shape}")
    out = phone.copy()
    for level_offset, burden_offset in ((0, 6), (10, 9)):
        levels = np.abs(paired[level_offset:level_offset + 4]).astype(np.float32)
        gates = np.asarray(
            (
                levels[2] / 40.0,
                levels[3] / 40.0,
                np.mean(levels[0:2]) / 40.0,
            ),
            dtype=np.float32,
        )
        gates = np.clip(gates, 0.0, 1.0)
        out[burden_offset:burden_offset + 3] *= 1.0 + gates
    if not np.isfinite(out).all():
        raise ValueError("Non-finite level-conditioned phone features")
    return out


def alignment_feature_sets(ref, hyp, classifier=phone_class):
    """Return overall rate, class burden, and within-class conditional rates."""
    n, m = len(ref), len(hyp)
    if not n:
        raise ValueError("Reference lyric has no phones")
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, prev = dp[i], dp[i - 1]
        rp = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prev[j - 1] + (rp != hyp[j - 1]),
                prev[j] + 1,
                row[j - 1] + 1,
            )

    class_errors = [0, 0, 0]
    class_totals = [0, 0, 0]
    for phone in ref:
        group = classifier(phone)
        if group is not None:
            class_totals[group] += 1
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                group = classifier(ref[i - 1])
                if group is not None:
                    class_errors[group] += 1
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            group = classifier(ref[i - 1])
            if group is not None:
                class_errors[group] += 1
            i -= 1
        else:
            if not j or dp[i][j] != dp[i][j - 1] + 1:
                raise RuntimeError("Phone edit traceback is inconsistent")
            j -= 1
    overall = float(min(1.0, dp[n][m] / n))
    burdens = tuple(err / n for err in class_errors)
    conditional = tuple(
        err / total if total else 0.0
        for err, total in zip(class_errors, class_totals)
    )
    return overall, burdens, conditional


def alignment_error_type_features(ref, hyp, classifier=strict_phone_class):
    """Split reference-phone errors into deletions and substitutions.

    Insertions have no reference phone class, so only their overall rate is
    returned. Traceback priority matches ``alignment_feature_sets``.
    """
    n, m = len(ref), len(hyp)
    if not n:
        raise ValueError("Reference lyric has no phones")
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, prev = dp[i], dp[i - 1]
        rp = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prev[j - 1] + (rp != hyp[j - 1]),
                prev[j] + 1,
                row[j - 1] + 1,
            )

    deletion_by_class = [0, 0, 0]
    substitution_by_class = [0, 0, 0]
    deletions = substitutions = insertions = 0
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                substitutions += 1
                group = classifier(ref[i - 1])
                if group is not None:
                    substitution_by_class[group] += 1
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            group = classifier(ref[i - 1])
            if group is not None:
                deletion_by_class[group] += 1
            i -= 1
        else:
            if not j or dp[i][j] != dp[i][j - 1] + 1:
                raise RuntimeError("Phone edit traceback is inconsistent")
            insertions += 1
            j -= 1

    if deletions + substitutions + insertions != dp[n][m]:
        raise RuntimeError("Phone edit operation counts do not match distance")
    scale = float(n)
    return (
        float(min(1.0, dp[n][m] / scale)),
        (deletions / scale, substitutions / scale, insertions / scale),
        tuple(value / scale for value in deletion_by_class),
        tuple(value / scale for value in substitution_by_class),
    )


def alignment_labeled_burdens(ref, hyp, ref_groups, n_groups=3):
    """Return overall PER and error burdens for labels attached to reference positions."""
    n, m = len(ref), len(hyp)
    if not n or len(ref_groups) != n:
        raise ValueError("Reference phones and position labels must be nonempty and aligned")
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, prev = dp[i], dp[i - 1]
        rp = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prev[j - 1] + (rp != hyp[j - 1]),
                prev[j] + 1,
                row[j - 1] + 1,
            )
    errors = [0] * n_groups
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                group = ref_groups[i - 1]
                if group is not None:
                    errors[group] += 1
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            group = ref_groups[i - 1]
            if group is not None:
                errors[group] += 1
            i -= 1
        else:
            if not j or dp[i][j] != dp[i][j - 1] + 1:
                raise RuntimeError("Phone edit traceback is inconsistent")
            j -= 1
    return float(min(1.0, dp[n][m] / n)), tuple(value / n for value in errors)


def alignment_reference_error_mask(ref, hyp):
    """Return overall PER and a deterministic error flag for each reference phone.

    Substitutions and deletions mark their corresponding reference position.
    Insertions contribute to overall PER but cannot be assigned to a reference
    position.  The traceback priority intentionally matches
    ``alignment_feature_sets`` so the new joint features remain comparable to
    the existing strict burdens.
    """
    n, m = len(ref), len(hyp)
    if not n:
        raise ValueError("Reference lyric has no phones")
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, prev = dp[i], dp[i - 1]
        rp = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prev[j - 1] + (rp != hyp[j - 1]),
                prev[j] + 1,
                row[j - 1] + 1,
            )

    error_mask = np.zeros(n, dtype=np.bool_)
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                error_mask[i - 1] = True
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            error_mask[i - 1] = True
            i -= 1
        else:
            if not j or dp[i][j] != dp[i][j - 1] + 1:
                raise RuntimeError("Phone edit traceback is inconsistent")
            j -= 1
    if int(error_mask.sum()) > dp[n][m]:
        raise RuntimeError("Reference error mask exceeds edit distance")
    return float(min(1.0, dp[n][m] / n)), error_mask


def strict_joint_error_vector(ref, left, right):
    """Summarize errors shared by both ears versus errors unique to one ear."""
    left_rate, left_error = alignment_reference_error_mask(ref, left)
    right_rate, right_error = alignment_reference_error_mask(ref, right)
    groups = np.asarray(
        [strict_phone_class(phone) if strict_phone_class(phone) is not None else -1 for phone in ref],
        dtype=np.int8,
    )
    n_ref = float(len(ref))
    totals = np.asarray([(groups == group).sum() for group in range(3)], dtype=np.float64)
    shared = left_error & right_error
    complementary = left_error ^ right_error
    shared_burdens = np.asarray(
        [(shared & (groups == group)).sum() / n_ref for group in range(3)],
        dtype=np.float64,
    )
    complementary_burdens = np.asarray(
        [(complementary & (groups == group)).sum() / n_ref for group in range(3)],
        dtype=np.float64,
    )
    vector = np.asarray(
        (
            np.log1p(len(ref)),
            *(totals / n_ref),
            left_rate,
            right_rate,
            *shared_burdens,
            *complementary_burdens,
        ),
        dtype=np.float32,
    )
    if vector.shape != (12,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid strict joint error vector: shape={vector.shape}")
    if not ((0.0 <= vector[1:]) & (vector[1:] <= 1.0)).all():
        raise ValueError("Strict joint rates outside [0,1]")
    return vector


def alignment_features(ref, hyp):
    """Compatibility view used by the original burden experiments."""
    overall, burdens, _ = alignment_feature_sets(ref, hyp)
    return overall, burdens


def check_alignment():
    assert alignment_features(("p", "æ", "t"), ("p", "æ", "t")) == (0.0, (0.0, 0.0, 0.0))
    assert alignment_features(("p", "æ", "t"), ("b", "æ", "t"))[1][0] == 1 / 3
    assert alignment_features(("s", "æ"), ("æ",))[1][1] == 1 / 2
    assert alignment_features(("æ",), ("b", "æ"))[1] == (0.0, 0.0, 0.0)
    overall, burden, conditional = alignment_feature_sets(("p", "æ", "t"), ("b", "æ", "t"))
    assert overall == 1 / 3 and burden[0] == 1 / 3 and conditional[0] == 1 / 2
    assert conditional[1:] == (0.0, 0.0)
    assert strict_phone_class("t") == 0
    assert strict_phone_class("tʃ") == 1
    assert strict_phone_class("dʒ") == 1
    assert strict_phone_class("oɹ") is None
    assert strict_phone_class("ɹ") == 2
    assert manner3_phone_class("b") == 0
    assert manner3_phone_class("dʒ") == 0
    assert manner3_phone_class("ʃ") == 1
    assert manner3_phone_class("ɑɹ") is None
    assert [top5_phone_class(phone) for phone in ("t", "n", "ɪ", "s", "l", "ɑ")] == [
        0, 1, 2, 3, 4, 5
    ]
    typed = alignment_error_type_features(
        ("p", "æ", "t"), ("b", "æ"), classifier=strict_phone_class
    )
    assert typed[0] == 2 / 3
    assert typed[1] == (1 / 3, 1 / 3, 0.0)
    assert typed[2] == (1 / 3, 0.0, 0.0)
    assert typed[3] == (1 / 3, 0.0, 0.0)
    positioned = alignment_labeled_burdens(
        ("p", "æ", "t"), ("b", "æ"), (0, None, 2)
    )
    assert positioned == (2 / 3, (1 / 3, 0.0, 1 / 3))
    joint = strict_joint_error_vector(
        ("p", "æ", "t"), ("b", "æ", "t"), ("p", "æ")
    )
    assert np.isclose(joint[4], 1 / 3) and np.isclose(joint[5], 1 / 3)
    assert np.isclose(joint[6:9].sum(), 0.0)
    assert np.isclose(joint[9], 2 / 3)
    _, strict_burden, _ = alignment_feature_sets(
        ("tʃ", "oɹ", "ɹ"), ("s", "oɹ", "l"), classifier=strict_phone_class
    )
    assert strict_burden == (0.0, 1 / 3, 1 / 3)
    symmetric = strict_be_gap_vector((0.4, 0.3, 0.1, 0.2, 0.3, 0.05, 0.25, 0.1))
    assert np.allclose(symmetric, (0.3, 0.1, 0.05, 0.05, 0.2, 0.05, 0.1, 0.2))


def load_inputs(trainer, cfg):
    paths = cfg["paths"]
    train_ids, dev_ids = trainer.load_split_ids(paths["split_dir"])
    metadata = trainer.load_metadata(paths["metadata"])
    records = (
        trainer.select_records_by_ids(metadata, train_ids)
        + trainer.select_records_by_ids(metadata, dev_ids)
    )
    if len(train_ids) != 16238 or len(dev_ids) != 4059 or len(records) != 20297:
        raise ValueError("Unexpected fixed by-prompt train/dev split")
    prompts = {str(row["signal"]): str(row["prompt"]) for row in records}
    if len(prompts) != len(records):
        raise ValueError("Duplicate signal ID in metadata")

    asr = {}
    path = Path(paths["confidence_v3_jsonl"])
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            sid = str(item.get("signal", ""))
            if not sid or sid in asr:
                raise ValueError(f"Invalid or duplicate signal at {path}:{line_no}")
            if sid not in prompts:
                continue
            asr[sid] = (
                str((item.get("left") or {}).get("text", "")),
                str((item.get("right") or {}).get("text", "")),
            )
    if set(asr) != set(prompts):
        raise ValueError(f"V3 ASR text coverage mismatch: missing {len(set(prompts) - set(asr))}")

    digest = hashlib.sha256()
    for sid in sorted(prompts):
        digest.update(
            (json.dumps([sid, prompts[sid], *asr[sid]], ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        )
    return prompts, asr, digest.hexdigest()


def espeak_version():
    proc = subprocess.run(["espeak-ng", "--version"], check=True, capture_output=True, text=True)
    return proc.stdout.splitlines()[0]


def save_word_cache(words, version):
    tmp = WORD_CACHE.with_suffix(".json.tmp")
    payload = {"schema_version": 1, "espeak_version": version, "words": words}
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, WORD_CACHE)


def load_or_build_words(prompts, asr, version):
    if WORD_CACHE.exists():
        cache = json.loads(WORD_CACHE.read_text(encoding="utf-8"))
        if cache.get("schema_version") != 1 or cache.get("espeak_version") != version:
            raise ValueError(f"Incompatible eSpeak word cache: {WORD_CACHE}")
        words = cache.get("words")
        if not isinstance(words, dict):
            raise ValueError("Malformed eSpeak word cache")
    else:
        words = {}
    vocabulary = set()
    for sid, prompt in prompts.items():
        for phrase in (prompt, *asr[sid]):
            vocabulary.update(normalized_words(phrase))
    missing = sorted(vocabulary - words.keys())
    print(f"IPA vocabulary: {len(vocabulary):,} words | cached={len(vocabulary)-len(missing):,} | new={len(missing):,}", flush=True)
    for i, word in enumerate(tqdm(missing, desc="Phonemizing unique words", unit="word", dynamic_ncols=True), 1):
        words[word] = list(phonemize_word(word))
        if i % 250 == 0:
            save_word_cache(words, version)
    if missing:
        save_word_cache(words, version)
    return words


def phrase_phones(text, word_phones):
    return tuple(phone for word in normalized_words(text) for phone in word_phones[word])


def phrase_phones_with_word_position(text, word_phones):
    """Return phones plus onset/medial/coda labels for reference consonants."""
    phones = []
    groups = []
    for word in normalized_words(text):
        sequence = tuple(word_phones[word])
        vowel_indices = [
            index for index, phone in enumerate(sequence)
            if phone not in EXACT_CONSONANTS
        ]
        first_vowel = min(vowel_indices) if vowel_indices else None
        last_vowel = max(vowel_indices) if vowel_indices else None
        for index, phone in enumerate(sequence):
            phones.append(phone)
            if phone not in EXACT_CONSONANTS:
                groups.append(None)
            elif first_vowel is None:
                groups.append(1)
            elif index < first_vowel:
                groups.append(0)
            elif index > last_vowel:
                groups.append(2)
            else:
                groups.append(1)
    return tuple(phones), tuple(groups)


def load_feature_cache(required_ids, digest, version):
    if not FEATURE_CACHE.exists() and not FEATURE_MANIFEST.exists():
        return None
    if not FEATURE_CACHE.exists() or not FEATURE_MANIFEST.exists():
        raise ValueError("Incomplete phone-error feature cache; inspect before retrying")
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "input_sha256": digest, "espeak_version": version,
                "feature_names": list(PHONE_FEATURE_NAMES), "n_samples": 20297}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Phone-error feature cache mismatch: {key}")
    table = {}
    with FEATURE_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (8,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached phone features: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Phone-error feature cache ID coverage mismatch")
    print(f"Reusing {len(table):,} phone-error vectors: {FEATURE_CACHE}", flush=True)
    return table


def build_feature_cache(prompts, asr, digest, version):
    cached = load_feature_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    empty_left = empty_right = 0
    for sid in tqdm(sorted(prompts), desc="Aligning reference and ASR phones", unit="sample", dynamic_ncols=True):
        ref = phrase_phones(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        empty_left += not bool(left)
        empty_right += not bool(right)
        left_rate, left_classes = alignment_features(ref, left)
        right_rate, right_classes = alignment_features(ref, right)
        vec = np.asarray((left_rate, right_rate, *left_classes, *right_classes), dtype=np.float32)
        if vec.shape != (8,) or not np.isfinite(vec).all() or not ((0 <= vec) & (vec <= 1)).all():
            raise ValueError(f"Invalid phone-error vector: {sid}")
        table[sid] = vec
    temp = FEATURE_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, FEATURE_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(PHONE_FEATURE_NAMES),
        "n_samples": len(table),
        "empty_left_asr": empty_left,
        "empty_right_asr": empty_right,
        "definition": "unit-cost global IPA-phone edit path; overall errors capped at 1; class errors per reference phone",
        "labels_used": False,
    }
    FEATURE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PHONE_CACHE_READY | {len(table):,} samples | empty ASR left={empty_left}, right={empty_right}", flush=True)
    return table


def load_strict_feature_cache(required_ids, digest, version):
    if not STRICT_FEATURE_CACHE.exists() and not STRICT_FEATURE_MANIFEST.exists():
        return None
    if not STRICT_FEATURE_CACHE.exists() or not STRICT_FEATURE_MANIFEST.exists():
        raise ValueError("Incomplete strict phone feature cache; inspect before retrying")
    manifest = json.loads(STRICT_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(STRICT_FEATURE_NAMES),
        "n_samples": 20297,
        "classifier": "exact_token_strict3_v1",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Strict phone feature cache mismatch: {key}")
    table = {}
    with STRICT_FEATURE_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (8,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached strict phone features: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Strict phone feature cache ID coverage mismatch")
    print(f"Reusing {len(table):,} strict phone-error vectors: {STRICT_FEATURE_CACHE}", flush=True)
    return table


def build_strict_feature_cache(prompts, asr, digest, version):
    cached = load_strict_feature_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    empty_left = empty_right = 0
    for sid in tqdm(
        sorted(prompts), desc="Aligning strict reference and ASR phones",
        unit="sample", dynamic_ncols=True,
    ):
        ref = phrase_phones(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        empty_left += not bool(left)
        empty_right += not bool(right)
        left_rate, left_burdens, _ = alignment_feature_sets(
            ref, left, classifier=strict_phone_class
        )
        right_rate, right_burdens, _ = alignment_feature_sets(
            ref, right, classifier=strict_phone_class
        )
        vec = np.asarray(
            (left_rate, right_rate, *left_burdens, *right_burdens),
            dtype=np.float32,
        )
        if vec.shape != (8,) or not np.isfinite(vec).all() or not ((0 <= vec) & (vec <= 1)).all():
            raise ValueError(f"Invalid strict phone feature vector: {sid}")
        table[sid] = vec
    temp = STRICT_FEATURE_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, STRICT_FEATURE_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(STRICT_FEATURE_NAMES),
        "n_samples": len(table),
        "empty_left_asr": empty_left,
        "empty_right_asr": empty_right,
        "classifier": "exact_token_strict3_v1",
        "strict_classes": {
            "voiceless_stop": sorted(STRICT_STOP),
            "fricative_affricate": sorted(STRICT_FRICATIVE_AFFRICATE),
            "sonorant_consonant": sorted(STRICT_SONORANT),
        },
        "definition": (
            "unit-cost global IPA-token edit path; exact-token strict classes; "
            "overall errors capped at 1; class errors per reference phone"
        ),
        "labels_used": False,
    }
    STRICT_FEATURE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"STRICT_PHONE_CACHE_READY | {len(table):,} samples | "
        f"empty ASR left={empty_left}, right={empty_right}",
        flush=True,
    )
    return table


def load_strict_joint_cache(required_ids, digest, version):
    if not STRICT_JOINT_CACHE.exists() and not STRICT_JOINT_MANIFEST.exists():
        return None
    if not STRICT_JOINT_CACHE.exists() or not STRICT_JOINT_MANIFEST.exists():
        raise ValueError("Incomplete strict joint-error cache; inspect before retrying")
    manifest = json.loads(STRICT_JOINT_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(STRICT_JOINT_FEATURE_NAMES),
        "n_samples": 20297,
        "classifier": "exact_token_strict3_v1",
        "traceback_priority": "match_or_substitution_then_deletion_then_insertion",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Strict joint-error cache mismatch: {key}")
    table = {}
    with STRICT_JOINT_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (12,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached strict joint-error features: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Strict joint-error cache ID coverage mismatch")
    print(
        f"Reusing {len(table):,} strict binaural joint-error vectors: {STRICT_JOINT_CACHE}",
        flush=True,
    )
    return table


def build_strict_joint_cache(prompts, asr, digest, version):
    cached = load_strict_joint_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    empty_left = empty_right = 0
    for sid in tqdm(
        sorted(prompts), desc="Aligning binaural joint reference-phone errors",
        unit="sample", dynamic_ncols=True,
    ):
        ref = phrase_phones(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        empty_left += not bool(left)
        empty_right += not bool(right)
        table[sid] = strict_joint_error_vector(ref, left, right)

    temp = STRICT_JOINT_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(
                json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n"
            )
    os.replace(temp, STRICT_JOINT_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(STRICT_JOINT_FEATURE_NAMES),
        "n_samples": len(table),
        "empty_left_asr": empty_left,
        "empty_right_asr": empty_right,
        "classifier": "exact_token_strict3_v1",
        "traceback_priority": "match_or_substitution_then_deletion_then_insertion",
        "definition": (
            "independent deterministic reference-to-left/right IPA alignments; "
            "substitution/deletion flags are compared at each reference position; "
            "class burdens divide shared-error and one-ear-only error counts by "
            "the total number of reference phones; insertions affect per-ear PER "
            "but have no reference position"
        ),
        "labels_used": False,
    }
    STRICT_JOINT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"STRICT_JOINT_CACHE_READY | {len(table):,} samples | "
        f"empty ASR left={empty_left}, right={empty_right}",
        flush=True,
    )
    return table


def load_strict_factorized_cache(required_ids, digest, version, alpha):
    if not STRICT_FACTORIZED_CACHE.exists() and not STRICT_FACTORIZED_MANIFEST.exists():
        return None
    if not STRICT_FACTORIZED_CACHE.exists() or not STRICT_FACTORIZED_MANIFEST.exists():
        raise ValueError("Incomplete strict factorized cache; inspect before retrying")
    manifest = json.loads(STRICT_FACTORIZED_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(STRICT_FACTORIZED_FEATURE_NAMES),
        "n_samples": 20297,
        "classifier": "exact_token_strict3_v1",
        "smoothing_alpha": float(alpha),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Strict factorized cache mismatch: {key}")
    table = {}
    with STRICT_FACTORIZED_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (12,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid strict factorized vector: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Strict factorized cache ID coverage mismatch")
    print(
        f"Reusing {len(table):,} strict factorized vectors: {STRICT_FACTORIZED_CACHE}",
        flush=True,
    )
    return table


def build_strict_factorized_cache(prompts, asr, strict_raw, digest, version, alpha=3.0):
    cached = load_strict_factorized_cache(set(prompts), digest, version, alpha)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    for sid in tqdm(
        sorted(prompts), desc="Deriving strict smoothed phone features",
        unit="sample", dynamic_ncols=True,
    ):
        ref = phrase_phones(prompts[sid], words)
        n_ref = len(ref)
        if not n_ref:
            raise ValueError(f"Reference lyric has no phones: {sid}")
        totals = np.zeros(3, dtype=np.float64)
        for phone in ref:
            group = strict_phone_class(phone)
            if group is not None:
                totals[group] += 1.0
        raw = np.asarray(strict_raw[sid], dtype=np.float64)
        left_overall, right_overall = float(raw[0]), float(raw[1])
        # Strict raw burdens are error_count / n_ref. Float32 roundoff is far
        # below half a count for these short utterances, so nearest-integer
        # recovery is deterministic and avoids repeating the edit alignment.
        left_errors = np.rint(raw[2:5] * n_ref)
        right_errors = np.rint(raw[5:8] * n_ref)
        left_rates = (left_errors + alpha * left_overall) / (totals + alpha)
        right_rates = (right_errors + alpha * right_overall) / (totals + alpha)
        vec = np.asarray(
            (
                left_overall,
                right_overall,
                np.log1p(n_ref),
                *(totals / n_ref),
                *left_rates,
                *right_rates,
            ),
            dtype=np.float32,
        )
        if vec.shape != (12,) or not np.isfinite(vec).all():
            raise ValueError(f"Invalid strict factorized feature vector: {sid}")
        if not ((0 <= vec[[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]) &
                (vec[[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]] <= 1)).all():
            raise ValueError(f"Strict factorized rate outside [0,1]: {sid}")
        table[sid] = vec
    temp = STRICT_FACTORIZED_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, STRICT_FACTORIZED_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(STRICT_FACTORIZED_FEATURE_NAMES),
        "n_samples": len(table),
        "classifier": "exact_token_strict3_v1",
        "smoothing_alpha": float(alpha),
        "definition": (
            "strict reference class shares plus per-ear conditional error rates; "
            "rates are empirical-Bayes style shrinkage toward the same ear overall PER"
        ),
        "derived_from": str(STRICT_FEATURE_CACHE),
        "labels_used": False,
    }
    STRICT_FACTORIZED_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"STRICT_FACTORIZED_CACHE_READY | {len(table):,} samples", flush=True)
    return table


def load_manner3_cache(required_ids, digest, version):
    if not MANNER3_CACHE.exists() and not MANNER3_MANIFEST.exists():
        return None
    if not MANNER3_CACHE.exists() or not MANNER3_MANIFEST.exists():
        raise ValueError("Incomplete manner3 cache; inspect before retrying")
    manifest = json.loads(MANNER3_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(MANNER3_FEATURE_NAMES),
        "n_samples": 20297,
        "classifier": "exact_token_manner3_v1",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manner3 cache mismatch: {key}")
    table = {}
    with MANNER3_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (12,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached manner3 vector: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Manner3 cache ID coverage mismatch")
    print(f"Reusing {len(table):,} manner3 vectors: {MANNER3_CACHE}", flush=True)
    return table


def build_manner3_cache(prompts, asr, digest, version):
    cached = load_manner3_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    for sid in tqdm(
        sorted(prompts), desc="Aligning manner3 reference and ASR phones",
        unit="sample", dynamic_ncols=True,
    ):
        ref = phrase_phones(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        n_ref = len(ref)
        if not n_ref:
            raise ValueError(f"Reference lyric has no phones: {sid}")
        totals = np.zeros(3, dtype=np.float64)
        for phone in ref:
            group = manner3_phone_class(phone)
            if group is not None:
                totals[group] += 1.0
        left_overall, left_burdens, _ = alignment_feature_sets(
            ref, left, classifier=manner3_phone_class
        )
        right_overall, right_burdens, _ = alignment_feature_sets(
            ref, right, classifier=manner3_phone_class
        )
        vec = np.asarray(
            (
                np.log1p(n_ref),
                *(totals / n_ref),
                left_overall,
                right_overall,
                *left_burdens,
                *right_burdens,
            ),
            dtype=np.float32,
        )
        if vec.shape != (12,) or not np.isfinite(vec).all():
            raise ValueError(f"Invalid manner3 feature vector: {sid}")
        if not ((0 <= vec[1:]) & (vec[1:] <= 1)).all():
            raise ValueError(f"Manner3 rate outside [0,1]: {sid}")
        table[sid] = vec
    temp = MANNER3_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, MANNER3_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(MANNER3_FEATURE_NAMES),
        "n_samples": len(table),
        "classifier": "exact_token_manner3_v1",
        "classes": {
            "stop_affricate": sorted(MANNER3_STOP_AFFRICATE),
            "fricative": sorted(MANNER3_FRICATIVE),
            "sonorant_consonant": sorted(MANNER3_SONORANT),
        },
        "definition": (
            "reference demand and per-ear edit burdens for three exact-token manner classes"
        ),
        "labels_used": False,
    }
    MANNER3_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"MANNER3_CACHE_READY | {len(table):,} samples", flush=True)
    return table


def load_error_type_cache(required_ids, digest, version):
    if not ERROR_TYPE_CACHE.exists() and not ERROR_TYPE_MANIFEST.exists():
        return None
    if not ERROR_TYPE_CACHE.exists() or not ERROR_TYPE_MANIFEST.exists():
        raise ValueError("Incomplete strict error-type cache; inspect before retrying")
    manifest = json.loads(ERROR_TYPE_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(ERROR_TYPE_CACHE_FEATURE_NAMES),
        "n_samples": 20297,
        "classifier": "exact_token_strict3_v1",
        "traceback_priority": "match_or_substitution_then_deletion_then_insertion",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Strict error-type cache mismatch: {key}")
    table = {}
    with ERROR_TYPE_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (24,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached strict error-type vector: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Strict error-type cache ID coverage mismatch")
    print(f"Reusing {len(table):,} strict error-type vectors: {ERROR_TYPE_CACHE}", flush=True)
    return table


def build_error_type_cache(prompts, asr, digest, version):
    cached = load_error_type_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    for sid in tqdm(
        sorted(prompts), desc="Aligning strict deletion/substitution errors",
        unit="sample", dynamic_ncols=True,
    ):
        ref = phrase_phones(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        n_ref = len(ref)
        if not n_ref:
            raise ValueError(f"Reference lyric has no phones: {sid}")
        totals = np.zeros(3, dtype=np.float64)
        for phone in ref:
            group = strict_phone_class(phone)
            if group is not None:
                totals[group] += 1.0
        left_overall, left_rates, left_del, left_sub = alignment_error_type_features(
            ref, left, classifier=strict_phone_class
        )
        right_overall, right_rates, right_del, right_sub = alignment_error_type_features(
            ref, right, classifier=strict_phone_class
        )
        vec = np.asarray(
            (
                np.log1p(n_ref),
                *(totals / n_ref),
                left_overall,
                right_overall,
                *left_rates,
                *right_rates,
                *left_del,
                *left_sub,
                *right_del,
                *right_sub,
            ),
            dtype=np.float32,
        )
        if vec.shape != (24,) or not np.isfinite(vec).all() or (vec < 0).any():
            raise ValueError(f"Invalid strict error-type feature vector: {sid}")
        bounded = np.asarray((*range(1, 8), 9, 10, *range(12, 24)), dtype=np.int64)
        if not (vec[bounded] <= 1).all():
            raise ValueError(f"Strict error-type rate outside expected range: {sid}")
        table[sid] = vec
    temp = ERROR_TYPE_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, ERROR_TYPE_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(ERROR_TYPE_CACHE_FEATURE_NAMES),
        "n_samples": len(table),
        "classifier": "exact_token_strict3_v1",
        "traceback_priority": "match_or_substitution_then_deletion_then_insertion",
        "definition": (
            "strict reference demand plus deterministic Levenshtein deletion, "
            "substitution, and insertion rates; insertions have no reference class"
        ),
        "labels_used": False,
    }
    ERROR_TYPE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ERROR_TYPE_CACHE_READY | {len(table):,} samples", flush=True)
    return table


def deletion_emphasis_vectors(detailed, substitution_weight=0.5):
    """Return 12-D demand + overall PER + deletion-emphasized class burdens."""
    table = {}
    for sid, raw in detailed.items():
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (24,):
            raise ValueError(f"Expected 24 strict error-type features: {sid}")
        left_weighted = raw[12:15] + substitution_weight * raw[15:18]
        right_weighted = raw[18:21] + substitution_weight * raw[21:24]
        vec = np.concatenate((raw[0:6], left_weighted, right_weighted)).astype(np.float32)
        if vec.shape != (12,) or not np.isfinite(vec).all():
            raise ValueError(f"Invalid deletion-emphasis feature vector: {sid}")
        table[sid] = vec
    return table


def strict_dual_vectors(strict_raw, strict_factorized):
    """Combine demand, PER, raw burden, and smoothed conditional rates."""
    if set(strict_raw) != set(strict_factorized):
        raise ValueError("Strict raw/factorized cache ID mismatch")
    table = {}
    for sid in strict_raw:
        raw = np.asarray(strict_raw[sid], dtype=np.float32)
        factorized = np.asarray(strict_factorized[sid], dtype=np.float32)
        vec = np.concatenate(
            (
                factorized[2:6],
                raw[0:2],
                raw[2:8],
                factorized[6:12],
            )
        ).astype(np.float32)
        if vec.shape != (18,) or not np.isfinite(vec).all():
            raise ValueError(f"Invalid strict dual feature vector: {sid}")
        table[sid] = vec
    return table


def load_position_cache(required_ids, digest, version):
    if not POSITION_CACHE.exists() and not POSITION_MANIFEST.exists():
        return None
    if not POSITION_CACHE.exists() or not POSITION_MANIFEST.exists():
        raise ValueError("Incomplete phone-position cache; inspect before retrying")
    manifest = json.loads(POSITION_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(POSITION_FEATURE_NAMES),
        "n_samples": 20297,
        "position_definition": "before_first_vowel_between_vowels_after_last_vowel",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Phone-position cache mismatch: {key}")
    table = {}
    with POSITION_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (12,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached phone-position vector: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Phone-position cache ID coverage mismatch")
    print(f"Reusing {len(table):,} phone-position vectors: {POSITION_CACHE}", flush=True)
    return table


def build_position_cache(prompts, asr, digest, version):
    cached = load_position_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    for sid in tqdm(
        sorted(prompts), desc="Aligning onset/medial/coda consonant errors",
        unit="sample", dynamic_ncols=True,
    ):
        ref, ref_groups = phrase_phones_with_word_position(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        n_ref = len(ref)
        if not n_ref:
            raise ValueError(f"Reference lyric has no phones: {sid}")
        totals = np.zeros(3, dtype=np.float64)
        for group in ref_groups:
            if group is not None:
                totals[group] += 1.0
        left_overall, left_burdens = alignment_labeled_burdens(ref, left, ref_groups)
        right_overall, right_burdens = alignment_labeled_burdens(ref, right, ref_groups)
        vec = np.asarray(
            (
                np.log1p(n_ref),
                *(totals / n_ref),
                left_overall,
                right_overall,
                *left_burdens,
                *right_burdens,
            ),
            dtype=np.float32,
        )
        if vec.shape != (12,) or not np.isfinite(vec).all():
            raise ValueError(f"Invalid phone-position feature vector: {sid}")
        if not ((0 <= vec[1:]) & (vec[1:] <= 1)).all():
            raise ValueError(f"Phone-position rate outside [0,1]: {sid}")
        table[sid] = vec
    temp = POSITION_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, POSITION_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(POSITION_FEATURE_NAMES),
        "n_samples": len(table),
        "position_definition": "before_first_vowel_between_vowels_after_last_vowel",
        "consonant_inventory": sorted(EXACT_CONSONANTS),
        "definition": (
            "reference consonant demand and per-ear edit burdens by within-word position"
        ),
        "labels_used": False,
    }
    POSITION_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"POSITION_CACHE_READY | {len(table):,} samples", flush=True)
    return table


def load_top5_cache(required_ids, digest, version):
    if not TOP5_CACHE.exists() and not TOP5_MANIFEST.exists():
        return None
    if not TOP5_CACHE.exists() or not TOP5_MANIFEST.exists():
        raise ValueError("Incomplete top5-phone cache; inspect before retrying")
    manifest = json.loads(TOP5_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(TOP5_FEATURE_NAMES),
        "n_samples": 20297,
        "selected_phones": list(TOP5_PHONES),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Top5-phone cache mismatch: {key}")
    table = {}
    with TOP5_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (20,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached top5-phone vector: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Top5-phone cache ID coverage mismatch")
    print(f"Reusing {len(table):,} top5-phone vectors: {TOP5_CACHE}", flush=True)
    return table


def build_top5_cache(prompts, asr, digest, version):
    cached = load_top5_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    for sid in tqdm(
        sorted(prompts), desc="Aligning top5 exact-phone errors",
        unit="sample", dynamic_ncols=True,
    ):
        ref = phrase_phones(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        n_ref = len(ref)
        if not n_ref:
            raise ValueError(f"Reference lyric has no phones: {sid}")
        groups = tuple(top5_phone_class(phone) for phone in ref)
        totals = np.bincount(groups, minlength=6).astype(np.float64)
        left_overall, left_burdens = alignment_labeled_burdens(
            ref, left, groups, n_groups=6
        )
        right_overall, right_burdens = alignment_labeled_burdens(
            ref, right, groups, n_groups=6
        )
        vec = np.asarray(
            (
                left_overall,
                right_overall,
                *(totals / n_ref),
                *left_burdens,
                *right_burdens,
            ),
            dtype=np.float32,
        )
        if vec.shape != (20,) or not np.isfinite(vec).all():
            raise ValueError(f"Invalid top5-phone feature vector: {sid}")
        if not ((0 <= vec) & (vec <= 1)).all():
            raise ValueError(f"Top5-phone rate outside [0,1]: {sid}")
        if not np.isclose(vec[2:8].sum(), 1.0, atol=1e-6):
            raise ValueError(f"Top5-phone demand shares do not sum to one: {sid}")
        table[sid] = vec
    temp = TOP5_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, TOP5_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(TOP5_FEATURE_NAMES),
        "n_samples": len(table),
        "selected_phones": list(TOP5_PHONES),
        "selection_rule": "five most frequent normalized reference-phone tokens in corpus audit",
        "definition": (
            "reference demand and per-ear edit burdens for five frequent exact phones "
            "plus one exhaustive other-phone group"
        ),
        "labels_used": False,
    }
    TOP5_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"TOP5_CACHE_READY | {len(table):,} samples", flush=True)
    return table


def load_rate_feature_cache(required_ids, digest, version):
    if not RATE_FEATURE_CACHE.exists() and not RATE_FEATURE_MANIFEST.exists():
        return None
    if not RATE_FEATURE_CACHE.exists() or not RATE_FEATURE_MANIFEST.exists():
        raise ValueError("Incomplete conditional-rate feature cache; inspect before retrying")
    manifest = json.loads(RATE_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(RATE_FEATURE_NAMES),
        "n_samples": 20297,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Conditional-rate feature cache mismatch: {key}")
    table = {}
    with RATE_FEATURE_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["signal"])
            vec = np.asarray(item["features"], dtype=np.float32)
            if sid in table or vec.shape != (8,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid cached conditional-rate features: {sid}")
            table[sid] = vec
    if set(table) != required_ids:
        raise ValueError("Conditional-rate feature cache ID coverage mismatch")
    print(f"Reusing {len(table):,} conditional-rate vectors: {RATE_FEATURE_CACHE}", flush=True)
    return table


def build_rate_feature_cache(prompts, asr, digest, version):
    cached = load_rate_feature_cache(set(prompts), digest, version)
    if cached is not None:
        return cached
    words = load_or_build_words(prompts, asr, version)
    table = {}
    empty_left = empty_right = 0
    for sid in tqdm(sorted(prompts), desc="Building conditional phone-class rates", unit="sample", dynamic_ncols=True):
        ref = phrase_phones(prompts[sid], words)
        left_text, right_text = asr[sid]
        left = phrase_phones(left_text, words)
        right = phrase_phones(right_text, words)
        empty_left += not bool(left)
        empty_right += not bool(right)
        left_rate, _, left_conditional = alignment_feature_sets(ref, left)
        right_rate, _, right_conditional = alignment_feature_sets(ref, right)
        vec = np.asarray(
            (left_rate, right_rate, *left_conditional, *right_conditional),
            dtype=np.float32,
        )
        if vec.shape != (8,) or not np.isfinite(vec).all() or not ((0 <= vec) & (vec <= 1)).all():
            raise ValueError(f"Invalid conditional-rate vector: {sid}")
        table[sid] = vec
    temp = RATE_FEATURE_CACHE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for sid in sorted(table):
            stream.write(json.dumps({"signal": sid, "features": table[sid].tolist()}) + "\n")
    os.replace(temp, RATE_FEATURE_CACHE)
    manifest = {
        "schema_version": 1,
        "input_sha256": digest,
        "espeak_version": version,
        "feature_names": list(RATE_FEATURE_NAMES),
        "n_samples": len(table),
        "empty_left_asr": empty_left,
        "empty_right_asr": empty_right,
        "definition": "unit-cost global IPA-phone edit path; each class error count divided by the number of reference phones in that class; zero when class absent",
        "labels_used": False,
    }
    RATE_FEATURE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"RATE_CACHE_READY | {len(table):,} samples | "
        f"empty ASR left={empty_left}, right={empty_right}",
        flush=True,
    )
    return table


def describe_phone_features(table, feature_names):
    values = np.stack([table[sid] for sid in sorted(table)]).astype(np.float64)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature table/name width mismatch: table={values.shape}, "
            f"names={len(feature_names)}"
        )
    for col, name in enumerate(feature_names):
        print(
            f"  {name:37s} mean={values[:, col].mean():.4f} "
            f"std={values[:, col].std():.4f} "
            f"nonzero={np.mean(values[:, col] > 0):.1%}",
            flush=True,
        )


def mode_phone_spec(mode):
    if mode == "class_rate":
        return RATE_FEATURE_NAMES, RATE_FEATURE_CACHE
    if mode == "strict_raw":
        return STRICT_FEATURE_NAMES, STRICT_FEATURE_CACHE
    if mode == "strict_be_gap":
        return STRICT_BE_GAP_FEATURE_NAMES, STRICT_FEATURE_CACHE
    if mode in ("strict_smoothed", "strict_factorized"):
        return STRICT_FACTORIZED_FEATURE_NAMES, STRICT_FACTORIZED_CACHE
    if mode in (
        "strict_burden_demand",
        "strict_burden_demand_branch",
        "strict_burden_level_interaction",
    ):
        return STRICT_BURDEN_DEMAND_FEATURE_NAMES, STRICT_FACTORIZED_CACHE
    if mode == "manner3_burden_demand":
        return MANNER3_FEATURE_NAMES, MANNER3_CACHE
    if mode == "strict_deletion_emphasis_demand":
        return DELETION_EMPHASIS_FEATURE_NAMES, ERROR_TYPE_CACHE
    if mode == "position_burden_demand":
        return POSITION_FEATURE_NAMES, POSITION_CACHE
    if mode == "top5_phone_burden_demand":
        return TOP5_FEATURE_NAMES, TOP5_CACHE
    if mode == "strict_dual_burden_rate":
        return STRICT_DUAL_FEATURE_NAMES, STRICT_FACTORIZED_CACHE
    if mode == "strict_joint_error_demand":
        return STRICT_JOINT_FEATURE_NAMES, STRICT_JOINT_CACHE
    return PHONE_FEATURE_NAMES, FEATURE_CACHE


def mode_text_feature_names(mode):
    if mode == "consonant_lyrics":
        return PHONETIC_DEMAND_NAMES
    if mode in (
        "strict_factorized",
        "strict_burden_demand",
        "manner3_burden_demand",
        "strict_deletion_emphasis_demand",
        "position_burden_demand",
        "strict_burden_demand_branch",
        "strict_burden_level_interaction",
        "top5_phone_burden_demand",
        "strict_dual_burden_rate",
        "strict_joint_error_demand",
    ):
        return ()
    return TEXT_FEATURE_NAMES


def prepare_config(mode):
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if int(cfg["paired_feature"]["dim"]) != 20 or int(cfg["fusion"]["fused_dim"]) != 306:
        raise ValueError("Expected paired D6e 20-D inputs and 306-D fusion")
    cfg["experiment"] = {"id": "D6e_PhoneClassErrors_v1", "name": f"level_text_{mode}"}
    cfg["paths"]["output_root"] = str(HERE / "results")
    cfg["paired_feature"]["phone_error_mode"] = mode
    feature_names, feature_cache = mode_phone_spec(mode)
    cfg["paired_feature"]["phone_error_cache"] = str(feature_cache)
    if mode == "strict_be_gap":
        cfg["paired_feature"]["phone_feature_transform"] = (
            "per quantity: min(left,right), abs(left-right)"
        )
    if mode in ("strict_smoothed", "strict_factorized", "strict_dual_burden_rate"):
        cfg["paired_feature"]["strict_rate_smoothing_alpha"] = 3.0
        cfg["paired_feature"]["strict_rate_smoothing_prior"] = "same-ear overall phone error rate"
    if mode in (
        "strict_burden_demand",
        "strict_burden_demand_branch",
        "strict_burden_level_interaction",
        "strict_dual_burden_rate",
    ):
        cfg["paired_feature"]["strict_raw_cache"] = str(STRICT_FEATURE_CACHE)
        cfg["paired_feature"]["strict_factorized_cache"] = str(STRICT_FACTORIZED_CACHE)
    if mode == "manner3_burden_demand":
        cfg["paired_feature"]["phone_taxonomy"] = "exact_token_manner3_v1"
    if mode == "strict_deletion_emphasis_demand":
        cfg["paired_feature"]["phone_taxonomy"] = "exact_token_strict3_v1"
        cfg["paired_feature"]["class_error_mix"] = {
            "deletion_weight": 1.0,
            "substitution_weight": 0.5,
        }
    if mode == "position_burden_demand":
        cfg["paired_feature"]["phone_grouping"] = (
            "reference_word_onset_medial_coda_consonant_v1"
        )
    if mode == "strict_burden_demand_branch":
        cfg["fusion"]["phone_feature_integration"] = "separate_12_to_8_branch"
        cfg["model"]["phone_branch_paired_dim"] = 20
        cfg["model"]["phone_branch_input_dim"] = 12
        cfg["model"]["phone_branch_output_dim"] = 8
        cfg["model"]["phone_branch_internal_fused_dim"] = 302
        cfg["model"]["regression_hidden_dim"] = 129
        cfg["model"]["parameter_matching_reference"] = (
            "direct_concat_306_to_128; branch head has 88 fewer parameters"
        )
    if mode == "strict_burden_level_interaction":
        cfg["paired_feature"]["phone_level_interaction"] = {
            "voiceless_stop_band_hz": [1500, 3500],
            "fricative_affricate_band_hz": [3500, 7600],
            "sonorant_bands_hz": [[150, 500], [500, 1500]],
            "formula": "burden * (1 + abs(level_delta_db) / 40)",
        }
    if mode == "top5_phone_burden_demand":
        cfg["paired_feature"]["dim"] = 28
        cfg["paired_feature"]["expanded_layout"] = (
            "8 level features + 20 top5-phone demand/error features"
        )
        cfg["fusion"]["fused_dim"] = 314
        cfg["model"]["regression_hidden_dim"] = 125
        cfg["model"]["parameter_matching_reference"] = (
            "314_to_125 head has 76 more parameters than 306_to_128"
        )
    if mode == "strict_dual_burden_rate":
        cfg["paired_feature"]["dim"] = 26
        cfg["paired_feature"]["expanded_layout"] = (
            "8 level + 4 demand + 2 PER + 6 burden + 6 smoothed rate"
        )
        cfg["fusion"]["fused_dim"] = 312
        cfg["model"]["regression_hidden_dim"] = 126
        cfg["model"]["parameter_matching_reference"] = (
            "312_to_126 head has 140 more parameters than 306_to_128"
        )
    if mode == "strict_joint_error_demand":
        cfg["paired_feature"]["binaural_joint_error_definition"] = (
            "per-reference-phone shared error versus one-ear-only error; "
            "insertions retained only in per-ear overall PER"
        )
    text_feature_names = mode_text_feature_names(mode)
    cfg["paired_feature"]["generic_text_feature_names"] = list(text_feature_names)
    if mode == "consonant_lyrics":
        cfg["paired_feature"]["reference_text_mode"] = "phonetic_demand"
        cfg["paired_feature"]["phonetic_demand_cache"] = str(PHONETIC_DEMAND_CACHE)
    phone_indices = MODE_PHONE_INDICES[mode]
    n_extra = len(text_feature_names) + len(phone_indices)
    cfg["paired_feature"]["phone_feature_names"] = [feature_names[i] for i in phone_indices]
    cfg["paired_feature"]["active_slots"] = (
        list(range(28 if mode == "top5_phone_burden_demand" else 26))
        if mode in ("top5_phone_burden_demand", "strict_dual_burden_rate")
        else list(LEVEL_INDICES + EXTRA_INDICES[:n_extra])
    )
    config_path = HERE / f"config_{mode}.yaml"
    if config_path.exists():
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != cfg:
            raise ValueError(f"Existing config differs: {config_path}")
    else:
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg, config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=tuple(MODE_PHONE_INDICES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="64 train / 32 dev / 2 epoch pipeline check")
    args = parser.parse_args()
    check_alignment()
    helper = load_ablation_helper()
    trainer = helper.load_trainer()
    cfg, config_path = prepare_config(args.mode)
    prompts, asr, digest = load_inputs(trainer, cfg)
    phonetic_demand = (
        load_phonetic_demand(prompts) if args.mode == "consonant_lyrics" else None
    )
    version = espeak_version()
    feature_names, feature_cache = mode_phone_spec(args.mode)
    if args.mode == "class_rate":
        phone_features = build_rate_feature_cache(prompts, asr, digest, version)
    elif args.mode in ("strict_raw", "strict_be_gap"):
        strict_raw_features = build_strict_feature_cache(prompts, asr, digest, version)
        phone_features = (
            {sid: strict_be_gap_vector(vec) for sid, vec in strict_raw_features.items()}
            if args.mode == "strict_be_gap"
            else strict_raw_features
        )
    elif args.mode in (
        "strict_smoothed",
        "strict_factorized",
        "strict_burden_demand",
        "strict_burden_demand_branch",
        "strict_burden_level_interaction",
        "strict_dual_burden_rate",
    ):
        strict_raw_features = build_strict_feature_cache(prompts, asr, digest, version)
        strict_factorized_features = build_strict_factorized_cache(
            prompts, asr, strict_raw_features, digest, version, alpha=3.0
        )
        if args.mode == "strict_dual_burden_rate":
            phone_features = strict_dual_vectors(
                strict_raw_features, strict_factorized_features
            )
        elif args.mode in (
            "strict_burden_demand",
            "strict_burden_demand_branch",
            "strict_burden_level_interaction",
        ):
            phone_features = {
                sid: np.concatenate(
                    (strict_factorized_features[sid][2:6], strict_raw_features[sid])
                ).astype(np.float32)
                for sid in strict_raw_features
            }
        else:
            phone_features = strict_factorized_features
    elif args.mode == "manner3_burden_demand":
        phone_features = build_manner3_cache(prompts, asr, digest, version)
    elif args.mode == "strict_deletion_emphasis_demand":
        detailed = build_error_type_cache(prompts, asr, digest, version)
        phone_features = deletion_emphasis_vectors(detailed, substitution_weight=0.5)
    elif args.mode == "position_burden_demand":
        phone_features = build_position_cache(prompts, asr, digest, version)
    elif args.mode == "top5_phone_burden_demand":
        phone_features = build_top5_cache(prompts, asr, digest, version)
    elif args.mode == "strict_joint_error_demand":
        phone_features = build_strict_joint_cache(prompts, asr, digest, version)
    else:
        phone_features = build_feature_cache(prompts, asr, digest, version)
    describe_phone_features(phone_features, feature_names)
    if args.prepare_only:
        print("PREPARE_READY | no training started", flush=True)
        return

    pooled_by_id = helper.load_pool_cache(trainer, cfg)
    original_load_branch = trainer.load_branch
    original_load_paired = trainer.load_paired_feature_table
    original_load_candidate = trainer.load_candidate_scalar_table
    original_model_class = trainer.WhisperDualConfidenceWERRegressor
    original_paired_feature_dim = trainer.PAIRED_FEATURE_DIM
    original_root = str(Path(cfg["paths"]["whisper_feature_root"]).resolve())

    def cached_branch(root, signal_id, branch_cfg):
        if str(Path(root).resolve()) != original_root or branch_cfg != cfg["branches"]["whisper"]:
            raise ValueError("Whisper branch differs from validated pooled cache")
        return pooled_by_id[str(signal_id)]

    def phone_paired_table(path, required_ids, dim=20):
        if args.mode in ("top5_phone_burden_demand", "strict_dual_burden_rate"):
            expected_dim = 28 if args.mode == "top5_phone_burden_demand" else 26
            if int(dim) != expected_dim:
                raise ValueError(
                    f"{args.mode} expects {expected_dim} paired dimensions, got {dim}"
                )
            source = original_load_paired(path, required_ids, 20)
            table = {}
            for sid, raw in source.items():
                expanded = np.empty(expected_dim, dtype=np.float32)
                expanded[:8] = raw[list(LEVEL_INDICES)]
                expanded[8:] = phone_features[sid]
                if not np.isfinite(expanded).all():
                    raise ValueError(f"Invalid expanded {args.mode} vector: {sid}")
                table[sid] = expanded
            print(
                f"Phone-error mode={args.mode}: {len(table):,} samples, "
                f"8 level + {expected_dim - 8} phone slots",
                flush=True,
            )
            return table

        table = original_load_paired(path, required_ids, dim)
        phone_indices = MODE_PHONE_INDICES[args.mode]
        text_feature_names = mode_text_feature_names(args.mode)
        n_extra = len(text_feature_names) + len(phone_indices)
        for sid, original in table.items():
            raw = original.copy()
            if phonetic_demand is not None:
                text_vec = phonetic_demand[sid]
            elif text_feature_names:
                text_vec = generic_text_features(prompts[sid])
            else:
                text_vec = np.empty(0, dtype=np.float32)
            phone_vec = phone_features[sid]
            if args.mode == "strict_burden_level_interaction":
                phone_vec = level_interaction_vector(phone_vec, raw)
            extra = np.concatenate((text_vec, phone_vec[list(phone_indices)]))
            original[:] = 0.0
            original[list(LEVEL_INDICES)] = raw[list(LEVEL_INDICES)]
            original[list(EXTRA_INDICES[:n_extra])] = extra
        print(f"Phone-error mode={args.mode}: {len(table):,} samples, 8 level + {n_extra} extra slots", flush=True)
        return table

    trainer.load_branch = cached_branch
    trainer.load_paired_feature_table = phone_paired_table
    trainer.load_candidate_scalar_table = (
        lambda root, required_ids, field="candidate_max_overall":
        load_candidate_scalar_cache(required_ids, field=field)
    )
    if args.mode == "strict_burden_demand_branch":
        trainer.WhisperDualConfidenceWERRegressor = make_phone_branch_regressor(trainer)
    if args.mode in ("top5_phone_burden_demand", "strict_dual_burden_rate"):
        trainer.PAIRED_FEATURE_DIM = (
            28 if args.mode == "top5_phone_burden_demand" else 26
        )
    try:
        trainer.train(cfg, config_path, args.seed, smoke=args.smoke, overwrite=False)
    finally:
        trainer.load_branch = original_load_branch
        trainer.load_paired_feature_table = original_load_paired
        trainer.load_candidate_scalar_table = original_load_candidate
        trainer.WhisperDualConfidenceWERRegressor = original_model_class
        trainer.PAIRED_FEATURE_DIM = original_paired_feature_dim

    run_dir = HERE / "results" / f"D6e_PhoneClassErrors_v1_level_text_{args.mode}" / (
        f"smoke_seed{args.seed}" if args.smoke else f"seed{args.seed}"
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["paired_feature"]["phone_error_mode"] = args.mode
    summary["paired_feature"]["phone_error_cache"] = str(feature_cache)
    summary["paired_feature"]["generic_text_feature_names"] = list(
        mode_text_feature_names(args.mode)
    )
    if args.mode == "consonant_lyrics":
        summary["paired_feature"]["reference_text_mode"] = "phonetic_demand"
        summary["paired_feature"]["phonetic_demand_cache"] = str(PHONETIC_DEMAND_CACHE)
    summary["paired_feature"]["phone_feature_names"] = [
        feature_names[i] for i in MODE_PHONE_INDICES[args.mode]
    ]
    if args.mode == "strict_be_gap":
        summary["paired_feature"]["phone_feature_transform"] = (
            "per quantity: min(left,right), abs(left-right)"
        )
    if args.mode in ("strict_smoothed", "strict_factorized", "strict_dual_burden_rate"):
        summary["paired_feature"]["strict_rate_smoothing_alpha"] = 3.0
        summary["paired_feature"]["strict_rate_smoothing_prior"] = (
            "same-ear overall phone error rate"
        )
    if args.mode in (
        "strict_burden_demand",
        "strict_burden_demand_branch",
        "strict_burden_level_interaction",
        "strict_dual_burden_rate",
    ):
        summary["paired_feature"]["strict_raw_cache"] = str(STRICT_FEATURE_CACHE)
        summary["paired_feature"]["strict_factorized_cache"] = str(
            STRICT_FACTORIZED_CACHE
        )
    if args.mode == "strict_burden_demand_branch":
        summary["fusion"] = {
            "type": "separate_phone_branch_then_concat",
            "phone_input_dim": 12,
            "phone_output_dim": 8,
            "internal_fused_dim": 302,
            "regression_hidden_dim": 129,
            "parameter_matching_reference": (
                "direct_concat_306_to_128; branch head has 88 fewer parameters"
            ),
        }
    if args.mode == "strict_burden_level_interaction":
        summary["paired_feature"]["phone_level_interaction"] = {
            "voiceless_stop_band_hz": [1500, 3500],
            "fricative_affricate_band_hz": [3500, 7600],
            "sonorant_bands_hz": [[150, 500], [500, 1500]],
            "formula": "burden * (1 + abs(level_delta_db) / 40)",
        }
    if args.mode == "top5_phone_burden_demand":
        summary["paired_feature"]["expanded_layout"] = (
            "8 level features + 20 top5-phone demand/error features"
        )
        summary["paired_feature"]["selected_phones"] = list(TOP5_PHONES)
        summary["paired_feature"]["other_group"] = "all remaining normalized phones"
        summary["model_parameter_matching"] = (
            "314_to_125 head has 76 more parameters than 306_to_128"
        )
    if args.mode == "strict_dual_burden_rate":
        summary["paired_feature"]["expanded_layout"] = (
            "8 level + 4 demand + 2 PER + 6 burden + 6 smoothed rate"
        )
        summary["model_parameter_matching"] = (
            "312_to_126 head has 140 more parameters than 306_to_128"
        )
    if args.mode == "strict_joint_error_demand":
        summary["paired_feature"]["binaural_joint_error_definition"] = (
            "per-reference-phone shared error versus one-ear-only error; "
            "insertions retained only in per-ear overall PER"
        )
    if args.mode == "manner3_burden_demand":
        summary["paired_feature"]["phone_taxonomy"] = "exact_token_manner3_v1"
    if args.mode == "strict_deletion_emphasis_demand":
        summary["paired_feature"]["phone_taxonomy"] = "exact_token_strict3_v1"
        summary["paired_feature"]["class_error_mix"] = {
            "deletion_weight": 1.0,
            "substitution_weight": 0.5,
        }
    if args.mode == "position_burden_demand":
        summary["paired_feature"]["phone_grouping"] = (
            "reference_word_onset_medial_coda_consonant_v1"
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"PHONE_RESULT mode={args.mode} seed={args.seed} smoke={args.smoke} "
        f"epoch={summary['best_epoch']} RMSE={summary['best_dev_rmse']:.9f} "
        f"PCC={summary['best_dev_pcc']:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
