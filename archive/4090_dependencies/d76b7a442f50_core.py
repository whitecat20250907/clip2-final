"""Small dependency-light helpers for reference-token likelihood features."""
from __future__ import annotations

import hashlib

import numpy as np


FEATURE_NAMES = (
    "reference_token_nll_mean",
    "reference_token_nll_std",
    "reference_token_nll_p90",
    "reference_token_low_support_fraction_p_lt_0p1",
)


def ids_sha256(signal_ids):
    return hashlib.sha256("\n".join(map(str, signal_ids)).encode("utf-8")).hexdigest()


def summarize_nll(nll):
    values = np.asarray(nll, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Token NLL must be a nonempty finite vector")
    result = np.asarray(
        [
            values.mean(),
            values.std(),
            np.quantile(values, 0.9),
            np.mean(values > np.log(10.0)),
        ],
        dtype=np.float32,
    )
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid reference-token summary")
    return result


def validate_feature_array(array, expected_rows):
    if array.shape != (expected_rows, len(FEATURE_NAMES)):
        raise ValueError(f"Unexpected feature shape: {array.shape}")
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Reference-token feature cache must be finite float32")

