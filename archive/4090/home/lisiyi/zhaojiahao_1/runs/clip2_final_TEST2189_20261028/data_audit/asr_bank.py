"""Numpy-only feature-bank helpers shared by bank.py, extract_pooled.py and the tests.

1. A byte-for-byte port of yj's 29-D zero-gated confidence recipe
   (archive/upstream/clip2-main/analyze_confidence_deep.py, _binaural.py, _dual.py +
   clip2_next/wavlm_dataset.py::COMBO_SPECS) so the vector can be recomputed for the
   official validation set, where her pickle does not exist. On TRAIN the bank audit
   compares this port against her pickle row by row.
2. The Whisper frame-count formula used by every teammate's masked pooling.
3. Lazy loaders for zzy's archived 30-D confidence / 12-D perceptual modules.
Human responses are never read here.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import re
import sys

import numpy as np

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']")
WORD_START = "Ġ"  # BPE 'Ġ': token begins a new word
HOP = 160              # 16 kHz -> 100 mel frames/s; encoder output is 50 Hz
MAX_FRAMES = 1500      # 30 s window


def norm_words(text):
    """yj: lowercase, strip punctuation (keep apostrophes), split on whitespace."""
    text = _WS.sub(" ", text or "")
    text = _PUNCT.sub(" ", text)
    return text.lower().split()


def tokens_to_words(tokens):
    """yj: merge subword tokens into words; a token starting with 'Ġ' starts a new word."""
    words = []
    cur = None
    for tk in tokens:
        t = tk.get("t", "")
        if t.startswith(WORD_START):
            if cur is not None:
                words.append(cur)
            cur = {"chars": [t[1:]], "p": [tk.get("p", np.nan)], "m": [tk.get("m", np.nan)]}
        else:
            if cur is None:
                cur = {"chars": [t], "p": [tk.get("p", np.nan)], "m": [tk.get("m", np.nan)]}
            else:
                cur["chars"].append(t)
                cur["p"].append(tk.get("p", np.nan))
                cur["m"].append(tk.get("m", np.nan))
    if cur is not None:
        words.append(cur)
    out = []
    for w in words:
        text = norm_words("".join(w["chars"]))
        if not text:  # pure punctuation token, drop
            continue
        p = np.array(w["p"], dtype=np.float64)
        m = np.array(w["m"], dtype=np.float64)
        out.append({"text": text[0], "prob": float(np.mean(p)), "margin": float(np.mean(m)),
                    "min_margin": float(np.min(m)), "n_tokens": len(p)})
    return out


def align_words(ref, hyp):
    """yj: word-level edit-distance alignment; hyp is a list of dicts with 'text'."""
    r, h = len(ref), len(hyp)
    dp = np.zeros((r + 1, h + 1), dtype=np.int32)
    dp[:, 0] = np.arange(r + 1)
    dp[0, :] = np.arange(h + 1)
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1]["text"] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    ops = []
    i, j = r, h
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]["text"] and dp[i, j] == dp[i - 1, j - 1]:
            ops.append(("match", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("del", i - 1, None)); i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ops.append(("ins", None, j - 1)); j -= 1
        else:
            ops.append(("sub", i - 1, j - 1)); i -= 1; j -= 1
    return ops[::-1]


def ear_feats(tokens, ref_words):
    """yj: zero-on-error features of one ear plus the per-reference-word match flags."""
    hyp_words = tokens_to_words(tokens or [])
    ops = align_words(ref_words, hyp_words)
    n_gt = max(len(ref_words), 1)
    prob = np.zeros(n_gt); margin = np.zeros(n_gt)
    flags = np.zeros(n_gt, dtype=bool)
    n_match = n_sub = n_del = n_ins = 0
    for op, ri, hi in ops:
        if op == "match":
            w = hyp_words[hi]
            prob[ri] = w["prob"]; margin[ri] = w["margin"]; flags[ri] = True
            n_match += 1
        elif op == "sub":
            n_sub += 1
        elif op == "del":
            n_del += 1
        else:
            n_ins += 1
    feats = {"zero_prob_mean": float(prob.mean()), "zero_margin_mean": float(margin.mean()),
             "frac_correct": float(n_match / n_gt), "sub_rate": n_sub / n_gt,
             "del_rate": n_del / n_gt, "ins_rate": n_ins / n_gt}
    return feats, flags


def combos(L, R):
    d = {}
    for name, (a, b) in {"zp": (L["zero_prob_mean"], R["zero_prob_mean"]),
                         "zm": (L["zero_margin_mean"], R["zero_margin_mean"]),
                         "fc": (L["frac_correct"], R["frac_correct"])}.items():
        lo, hi = min(a, b), max(a, b)
        d[f"min_ear_{name}"] = lo
        d[f"mean_ear_{name}"] = 0.5 * (a + b)
        d[f"max_ear_{name}"] = hi
        d[f"diff_ear_{name}"] = hi - lo
    return d


def cross_version(key, a, b):
    lo, hi = min(a, b), max(a, b)
    return {f"min_{key}": lo, f"mean_{key}": 0.5 * (a + b), f"max_{key}": hi,
            f"diff_{key}": hi - lo, f"prod_{key}": a * b}


def agreement(fl3_l, fl3_r, fl2_l, fl2_r):
    def fracs(fa, fb):
        n = max(fa.size, 1)
        return (float(np.logical_and(fa, fb).sum()) / n,
                float(np.logical_and(fa, np.logical_not(fb)).sum()) / n,
                float(np.logical_and(np.logical_not(fa), fb).sum()) / n)
    b = 0.5 * (fracs(fl3_l, fl2_l)[0] + fracs(fl3_r, fl2_r)[0])
    a3 = 0.5 * (fracs(fl3_l, fl2_l)[1] + fracs(fl3_r, fl2_r)[1])
    a2 = 0.5 * (fracs(fl3_l, fl2_l)[2] + fracs(fl3_r, fl2_r)[2])
    return {"both_correct": b, "v3_only": a3, "v2_only": a2, "neither": 1.0 - b - a3 - a2}


# Exactly clip2_next/wavlm_dataset.py::COMBO_SPECS order (29 columns).
YJ29_FIELDS = tuple(
    [f"{agg}_ear_{fam}" for fam in ("zp", "zm", "fc") for agg in ("min", "mean", "max", "diff")]
    + ["sub_rate", "del_rate", "ins_rate"]
    + ["mean_zp", "prod_zp", "max_zp", "min_zp", "mean_fc", "prod_fc", "max_fc", "min_fc",
       "both_correct", "neither"]
    + ["v2_mean_ear_zp", "v2_mean_ear_fc", "hl_mild", "hl_moderate"])
assert len(YJ29_FIELDS) == 29


def _channels(record):
    if record is None or record.get("error") is not None or "left" not in record or "right" not in record:
        return None
    return record["left"].get("tokens") or [], record["right"].get("tokens") or []


def yj29_vector(v3_record, v2_record, prompt, hearing_loss):
    """29-D combo aux for one item. Missing/failed records reproduce yj's fillna(0) rows."""
    ref = norm_words(prompt)
    row = {name: 0.0 for name in YJ29_FIELDS}
    hl = str(hearing_loss)
    row["hl_mild"] = 1.0 if hl == "Mild" else 0.0
    row["hl_moderate"] = 1.0 if hl == "Moderate" else 0.0
    c3 = _channels(v3_record)
    if c3 is not None:
        L, fl3_l = ear_feats(c3[0], ref)
        R, fl3_r = ear_feats(c3[1], ref)
        row.update({k: v for k, v in combos(L, R).items() if k in row})
        for rate in ("sub_rate", "del_rate", "ins_rate"):
            row[rate] = 0.5 * (L[rate] + R[rate])  # resolve_aux: nanmean over _L/_R columns
        c2 = _channels(v2_record)
        if c2 is not None:
            L2, fl2_l = ear_feats(c2[0], ref)
            R2, fl2_r = ear_feats(c2[1], ref)
            zp3 = 0.5 * (L["zero_prob_mean"] + R["zero_prob_mean"])
            zp2 = 0.5 * (L2["zero_prob_mean"] + R2["zero_prob_mean"])
            fc3 = 0.5 * (L["frac_correct"] + R["frac_correct"])
            fc2 = 0.5 * (L2["frac_correct"] + R2["frac_correct"])
            row.update({k: v for k, v in cross_version("zp", zp3, zp2).items() if k in row})
            row.update({k: v for k, v in cross_version("fc", fc3, fc2).items() if k in row})
            row.update({k: v for k, v in agreement(fl3_l, fl3_r, fl2_l, fl2_r).items() if k in row})
            row["v2_mean_ear_zp"] = zp2
            row["v2_mean_ear_fc"] = fc2
    vec = np.array([row[name] for name in YJ29_FIELDS], dtype=np.float64)
    # yj: df.replace([inf, -inf], nan).fillna(0.0)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def real_frames(n_samples_16k, hop=HOP, max_frames=MAX_FRAMES):
    """yj/zzy: number of non-padding encoder frames for a 16 kHz waveform."""
    real_mel = int(n_samples_16k) // hop + 1
    real_tok = max(1, (real_mel - 1) // 2 + 1)
    return min(real_tok, max_frames)


def masked_mean_std(frames, real_tok, eps=1e-12):
    """Population mean/std over the first `real_tok` frames (zzy's SP statistics)."""
    x = np.asarray(frames, dtype=np.float32)[: int(real_tok)].astype(np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("frames must be a non-empty [T, D] array")
    mean = x.mean(0)
    var = np.clip((x * x).mean(0) - mean * mean, 0.0, None)
    return mean.astype(np.float32), np.sqrt(var + eps).astype(np.float32)


ARCHIVE = {
    "zzy_confidence": "archive/upstream/clip2_zzy/best_version_modules_zzy/ASR_Lyrics_Confidence_Code.py",
    "zzy_perceptual": "archive/upstream/clip2_zzy/best_version_modules_zzy/STOI_ESTOI_PESQ_DNSMOS_12D_Code.py",
}


def load_archive_module(repo_root, key):
    """Import a teammate's archived module in place (their code is never copied or edited)."""
    path = Path(repo_root) / ARCHIVE[key]
    if not path.is_file():
        # Accept the authorized shared checkout directly; never copy or modify it.
        path = Path(repo_root) / ARCHIVE[key].removeprefix('archive/upstream/')
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(f"fusion_v2_archive_{key}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module, path


def finite(array, name):
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {name}")
    return array
