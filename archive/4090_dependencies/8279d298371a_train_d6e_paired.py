from __future__ import annotations
import argparse, json, math, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path("/home/lisiyi/ICASSP2_2026")
if not PROJECT_ROOT.is_dir():
    raise FileNotFoundError(f"CLIP2 project root missing: {PROJECT_ROOT}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clarity.recipes.clip2.stage1_20260811.common.cache import load_cached_layers
from clarity.recipes.clip2.stage1_20260811.common.datasets import load_metadata, select_records_by_ids
from clarity.recipes.clip2.stage1_20260811.common.metrics import compute_regression_metrics
from clarity.recipes.clip2.stage1_20260811.common.split import load_split_ids
from clarity.recipes.clip2.stage1_20260811.common.utils import (
    environment_info, prepare_output_dir, save_history_csv, save_json, set_seed
)

SIDE_FIELDS = (
    "prob.mean", "prob.min", "prob.std",
    "margin.mean", "margin.min", "margin.std",
)

PAIRED_FEATURE_DIM = 20


def load_paired_feature_table(path, required_ids, dim=PAIRED_FEATURE_DIM):
    table = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            sid = str(item["signal"])
            if sid in table:
                raise ValueError(f"{path}:{line_no}: duplicate signal {sid}")
            values = np.asarray(item["features"], dtype=np.float32)
            if values.shape != (dim,) or not np.isfinite(values).all():
                raise ValueError(f"{path}:{line_no}: expected finite {dim}-D features")
            table[sid] = values
    missing = set(required_ids) - table.keys()
    if missing:
        raise ValueError(f"Paired feature coverage incomplete: {len(missing)} missing; examples={sorted(missing)[:5]}")
    return table

def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a YAML mapping.")
    return cfg

def validate_branch(name, c):
    if not c.get("encoder_frozen", True):
        raise ValueError(f"{name}: encoder_frozen must be true.")
    pooling = c["pooling"]
    if pooling not in ("temporal_mean", "temporal_mean_std"):
        raise ValueError(f"{name}: unsupported pooling {pooling}")
    expected = c["frame_dim"] if pooling == "temporal_mean" else 2 * c["frame_dim"]
    if int(c["pooled_dim"]) != int(expected):
        raise ValueError(f"{name}: pooled_dim should be {expected}, got {c['pooled_dim']}")

def pool_frames(frames, mode):
    x = frames.float()
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"Expected non-empty [T,D], got {tuple(x.shape)}")
    mean = x.mean(0)
    if mode == "temporal_mean":
        return mean
    return torch.cat([mean, x.std(0, unbiased=False)], dim=-1)

def load_branch(root, signal_id, c):
    layer = int(c["cache_layer"])
    cached = load_cached_layers(
        root, signal_id,
        cache_format=c["cache_format"],
        feature_dim=int(c["frame_dim"]),
        requested_layers=[layer],
    )
    if set(cached) != {layer}:
        raise RuntimeError(f"{signal_id}: expected layer {layer}, got {sorted(cached)}")
    x = cached[layer]
    if x.ndim != 2 or x.shape[1] != int(c["frame_dim"]):
        raise RuntimeError(f"{signal_id}: bad cache shape {tuple(x.shape)}")
    z = pool_frames(x, c["pooling"])
    if z.shape != (int(c["pooled_dim"]),) or not torch.isfinite(z).all():
        raise RuntimeError(f"{signal_id}: bad pooled feature")
    return z

def _finite_float(v, label):
    try:
        x = float(v)
    except Exception as e:
        raise ValueError(f"{label}: cannot convert {v!r} to float") from e
    if not math.isfinite(x):
        raise ValueError(f"{label}: non-finite value {x}")
    return x

def extract_channel_features(ch, sid, side):
    if not isinstance(ch, dict):
        raise ValueError(f"{sid}/{side}: missing channel object")
    agg = ch.get("agg")
    if not isinstance(agg, dict):
        raise ValueError(f"{sid}/{side}: missing agg")
    prob, margin = agg.get("prob"), agg.get("margin")
    if not isinstance(prob, dict) or not isinstance(margin, dict):
        raise ValueError(f"{sid}/{side}: missing prob/margin aggregate")
    vals = [
        _finite_float(prob.get("mean"), f"{sid}/{side}/prob.mean"),
        _finite_float(prob.get("min"), f"{sid}/{side}/prob.min"),
        _finite_float(prob.get("std"), f"{sid}/{side}/prob.std"),
        _finite_float(margin.get("mean"), f"{sid}/{side}/margin.mean"),
        _finite_float(margin.get("min"), f"{sid}/{side}/margin.min"),
        _finite_float(margin.get("std"), f"{sid}/{side}/margin.std"),
    ]
    return np.asarray(vals, dtype=np.float32)

def load_confidence_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Confidence JSONL not found: {path}")
    table = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
            sid = str(obj.get("signal", ""))
            if not sid:
                raise ValueError(f"{path}:{line_no}: missing signal")
            if sid in table:
                raise ValueError(f"{path}:{line_no}: duplicate signal {sid}")
            left = extract_channel_features(obj.get("left"), sid, "left")
            right = extract_channel_features(obj.get("right"), sid, "right")
            feat = np.concatenate([left, right]).astype(np.float32)
            expected_dim = 2 * len(SIDE_FIELDS)
            if feat.shape != (expected_dim,) or not np.isfinite(feat).all():
                raise ValueError(f"{sid}: invalid {expected_dim}-D confidence vector")
            table[sid] = feat
    if not table:
        raise RuntimeError(f"No confidence rows loaded from {path}")
    return table

def verify_confidence_coverage(conf, metadata, train_ids, dev_ids):
    meta_ids = {str(r["signal"]) for r in metadata}
    tr, dv = set(map(str, train_ids)), set(map(str, dev_ids))
    missing_tr = sorted(tr - conf.keys())
    missing_dv = sorted(dv - conf.keys())
    extra = sorted(conf.keys() - meta_ids)
    print(f"Confidence rows : {len(conf)}")
    print(f"Coverage train  : {len(tr)-len(missing_tr)}/{len(tr)}")
    print(f"Coverage dev    : {len(dv)-len(missing_dv)}/{len(dv)}")
    print(f"Extra conf rows : {len(extra)}")
    if missing_tr or missing_dv:
        msg = (
            f"Confidence coverage incomplete: missing train={len(missing_tr)}, "
            f"dev={len(missing_dv)}. Examples: {(missing_tr + missing_dv)[:10]}"
        )
        raise RuntimeError(msg)

def fit_side_normalizer(conf, train_records, eps=1e-6):
    # IMPORTANT: fit on canonical TRAIN only, never DEV.
    x = np.stack([conf[str(r["signal"])] for r in train_records], axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    near_constant = std < eps
    safe_std = std.copy()
    safe_std[near_constant] = 1.0
    return mean.astype(np.float32), safe_std.astype(np.float32), std.astype(np.float32), near_constant


def load_candidate_scalar_table(root, required_ids, field="candidate_max_overall"):
    """Load one precomputed scalar from <root>/<signal>.json::binaural_summary[field]."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Candidate feature root not found: {root}")
    table = {}
    missing = []
    for sid in sorted(set(map(str, required_ids))):
        p = root / f"{sid}.json"
        if not p.is_file():
            missing.append(sid)
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"{p}: invalid JSON") from e
        obj_sid = str(obj.get("signal_id", obj.get("signal", "")))
        if obj_sid and obj_sid != sid:
            raise ValueError(f"{p}: signal mismatch file={sid!r}, json={obj_sid!r}")
        bs = obj.get("binaural_summary")
        if not isinstance(bs, dict):
            raise ValueError(f"{p}: missing binaural_summary")
        if field not in bs:
            raise ValueError(f"{p}: missing binaural_summary.{field}")
        v = _finite_float(bs[field], f"{sid}/binaural_summary.{field}")
        table[sid] = np.float32(v)
    if missing:
        raise RuntimeError(
            f"Candidate feature coverage incomplete: missing={len(missing)} "
            f"examples={missing[:10]}"
        )
    return table


def fit_scalar_normalizer(table, train_records, eps=1e-6):
    """Fit z-score parameters on canonical TRAIN only for a scalar feature."""
    x = np.asarray(
        [float(table[str(r["signal"])]) for r in train_records],
        dtype=np.float64,
    )
    mean = float(x.mean())
    raw_std = float(x.std())
    std = 1.0 if raw_std < float(eps) else raw_std
    return np.float32(mean), np.float32(std), np.float32(raw_std), bool(raw_std < float(eps))


def normalize_scalar_table(table, mean, std):
    out = {}
    for sid, v in table.items():
        z = (float(v) - float(mean)) / float(std)
        if not math.isfinite(z):
            raise ValueError(f"{sid}: non-finite normalized candidate scalar")
        out[sid] = np.asarray([z], dtype=np.float32)
    return out

def _normalize_text_for_wer(text):
    import string
    s = str(text or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def _word_error_rate(ref_text, hyp_text):
    ref = _normalize_text_for_wer(ref_text).split()
    hyp = _normalize_text_for_wer(hyp_text).split()
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i]
        for j, hw in enumerate(hyp, 1):
            cur.append(min(
                cur[-1] + 1,          # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + (rw != hw),  # substitution
            ))
        prev = cur
    return float(prev[-1]) / float(len(ref))

def build_better_ear_wer_table(conf, metadata, clip_max=1.0):
    """Compute clipped min(left_WER, right_WER) using metadata prompt as reference
    and the already-loaded v3 confidence JSONL's left/right ASR text."""
    meta_prompt = {str(r["signal"]): str(r["prompt"]) for r in metadata}
    table = {}
    # conf here is feature table only, so this helper is called with a raw-text table.
    for sid, pair in conf.items():
        if sid not in meta_prompt:
            continue
        ref = meta_prompt[sid]
        lw = _word_error_rate(ref, pair["left"])
        rw = _word_error_rate(ref, pair["right"])
        v = min(lw, rw)
        if clip_max is not None:
            v = min(v, float(clip_max))
        if not math.isfinite(v):
            raise ValueError(f"{sid}: non-finite better-ear WER")
        table[sid] = np.float32(v)
    return table

def build_extra_wer_table(texts, metadata, mode="wer3", clip_max=1.0, source_name="extra"):
    """Build one configurable extra-ASR WER vector.

    Supported modes:
      better_ear -> [min(L, R)]
      gap        -> [abs(L - R)] after clipping L/R individually
      left_right -> [L, R]
      wer3       -> [L, R, min(L, R)]

    The gap definition intentionally matches the OOF probe:
    left/right WER are clipped first, then abs(L-R) is computed.
    """
    mode = str(mode).lower()
    if mode not in ("better_ear", "gap", "left_right", "wer3"):
        raise ValueError(f"Unsupported {source_name} WER mode={mode!r}")

    meta_prompt = {str(r["signal"]): str(r["prompt"]) for r in metadata}
    table = {}
    for sid, pair in texts.items():
        if sid not in meta_prompt:
            continue

        ref = meta_prompt[sid]
        lw = _word_error_rate(ref, pair["left"])
        rw = _word_error_rate(ref, pair["right"])

        # Match probe semantics exactly: clip each ear before BE/GAP derivation.
        if clip_max is not None:
            lw = min(lw, float(clip_max))
            rw = min(rw, float(clip_max))

        be = min(lw, rw)
        gap = abs(lw - rw)

        vals = np.asarray({
            "better_ear": [be],
            "gap": [gap],
            "left_right": [lw, rw],
            "wer3": [lw, rw, be],
        }[mode], dtype=np.float32)

        if not np.isfinite(vals).all():
            raise ValueError(f"{sid}: non-finite {source_name} WER ({mode})")
        table[sid] = vals

    return table

def load_confidence_jsonl_with_text(path):
    """Same strict 12-D v3 confidence extraction as D1a, plus left/right decoded text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Confidence JSONL not found: {path}")
    features, texts = {}, {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
            sid = str(obj.get("signal", ""))
            if not sid:
                raise ValueError(f"{path}:{line_no}: missing signal")
            if sid in features:
                raise ValueError(f"{path}:{line_no}: duplicate signal {sid}")
            left_obj, right_obj = obj.get("left"), obj.get("right")
            left = extract_channel_features(left_obj, sid, "left")
            right = extract_channel_features(right_obj, sid, "right")
            feat = np.concatenate([left, right]).astype(np.float32)
            expected_dim = 2 * len(SIDE_FIELDS)
            if feat.shape != (expected_dim,) or not np.isfinite(feat).all():
                raise ValueError(f"{sid}: invalid {expected_dim}-D confidence vector")
            features[sid] = feat
            texts[sid] = {
                "left": str(left_obj.get("text", "")) if isinstance(left_obj, dict) else "",
                "right": str(right_obj.get("text", "")) if isinstance(right_obj, dict) else "",
            }
    if not features:
        raise RuntimeError(f"No confidence rows loaded from {path}")
    return features, texts


def build_dataset(
    records,
    whisper_root,
    wc,
    conf_v3,
    mean_v3,
    std_v3,
    conf_v2,
    mean_v2,
    std_v2,
    wer,
    desc,
    extra_wer=None,
):
    n = len(records)
    w = torch.empty(n, int(wc["pooled_dim"]), dtype=torch.float32)
    side_dim = 2 * len(SIDE_FIELDS)
    c_v3 = torch.empty(n, side_dim, dtype=torch.float32)
    c_v2 = torch.empty(n, side_dim, dtype=torch.float32)
    z = torch.empty(n, 1, dtype=torch.float32)
    extra_wer_dim = 0 if extra_wer is None else int(np.asarray(next(iter(extra_wer.values()))).size)
    xwer = torch.empty(n, extra_wer_dim, dtype=torch.float32) if extra_wer is not None else None
    y = torch.empty(n, dtype=torch.float32)

    for i, r in tqdm(enumerate(records), total=n, desc=desc, unit="sample", dynamic_ncols=True):
        sid = str(r["signal"])
        w[i].copy_(load_branch(whisper_root, sid, wc))

        raw_v3 = conf_v3[sid]
        raw_v2 = conf_v2[sid]
        norm_v3 = (raw_v3 - mean_v3) / std_v3
        norm_v2 = (raw_v2 - mean_v2) / std_v2

        if not np.isfinite(norm_v3).all():
            raise RuntimeError(f"{sid}: non-finite normalized v3 confidence")
        if not np.isfinite(norm_v2).all():
            raise RuntimeError(f"{sid}: non-finite normalized v2 confidence")

        c_v3[i].copy_(torch.from_numpy(norm_v3.astype(np.float32)))
        c_v2[i].copy_(torch.from_numpy(norm_v2.astype(np.float32)))
        if sid not in wer:
            raise RuntimeError(f"{sid}: missing better-ear WER")
        z[i, 0] = float(wer[sid])
        if xwer is not None:
            if sid not in extra_wer:
                raise RuntimeError(f"{sid}: missing extra WER features")
            xwer[i].copy_(torch.from_numpy(np.asarray(extra_wer[sid], dtype=np.float32)))

        target = float(r["correctness"])
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError(f"{sid}: invalid target {target}")
        y[i] = target

    return TensorDataset(w, c_v3, c_v2, z, y) if xwer is None else TensorDataset(w, c_v3, c_v2, z, xwer, y)

class Projection(nn.Module):
    def __init__(self, din, dout, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dout), nn.GELU(), nn.LayerNorm(dout), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class WhisperDualConfidenceWERRegressor(nn.Module):
    def __init__(self, wc, side_dim_v3, side_dim_v2, model_cfg, extra_wer_dim=0):
        super().__init__()
        d = float(model_cfg["dropout"])
        self.extra_wer_dim = int(extra_wer_dim)
        self.whisper_projection = Projection(wc["pooled_dim"], wc["projection_dim"], d)
        fused = int(wc["projection_dim"]) + int(side_dim_v3) + int(side_dim_v2) + 1 + self.extra_wer_dim
        h = int(model_cfg["regression_hidden_dim"])
        self.regressor = nn.Sequential(
            nn.Linear(fused, h), nn.GELU(), nn.Dropout(d), nn.Linear(h, 1), nn.Sigmoid()
        )
    def forward(self, w, c_v3, c_v2, z, extra_wer=None):
        w = self.whisper_projection(w)
        parts = [w, c_v3, c_v2, z]
        if self.extra_wer_dim:
            if extra_wer is None:
                raise ValueError("extra_wer is required when extra_wer_dim > 0")
            parts.append(extra_wer)
        return self.regressor(torch.cat(parts, dim=-1)).squeeze(-1)

def pearson_loss(pred, target, eps=1e-8):
    """Differentiable mini-batch Pearson loss: 1 - PCC."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = torch.sqrt(torch.sum(pred_c ** 2) * torch.sum(target_c ** 2) + eps)
    pcc = torch.sum(pred_c * target_c) / denom
    return 1.0 - pcc


def compute_training_loss(pred, target, loss_name, pcc_weight):
    mse = F.mse_loss(pred, target)
    if loss_name == "mse":
        return mse, mse.detach(), pred.new_tensor(0.0)
    if loss_name == "mse_pcc":
        ploss = pearson_loss(pred, target)
        return mse + pcc_weight * ploss, mse.detach(), ploss.detach()
    raise ValueError(f"Unsupported training.loss={loss_name!r}; expected 'mse' or 'mse_pcc'.")


def run_epoch(model, loader, device, optimizer=None, grad_clip=1.0, loss_name="mse", pcc_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss, total_mse_loss, total_pcc_loss, total_n, ps, ys = 0.0, 0.0, 0.0, 0, [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if len(batch) == 5:
                w, c_v3, c_v2, z, y = batch
                extra_wer = None
            elif len(batch) == 6:
                w, c_v3, c_v2, z, extra_wer, y = batch
            else:
                raise RuntimeError(f"Unexpected batch size {len(batch)}")
            w, c_v3, c_v2, z, y = w.to(device), c_v3.to(device), c_v2.to(device), z.to(device), y.to(device)
            if extra_wer is not None:
                extra_wer = extra_wer.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            p = model(w, c_v3, c_v2, z, extra_wer)
            loss, mse_component, pcc_component = compute_training_loss(p, y, loss_name, pcc_weight)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * y.shape[0]
            total_mse_loss += mse_component.item() * y.shape[0]
            total_pcc_loss += pcc_component.item() * y.shape[0]
            total_n += y.shape[0]
            ps.append(p.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    m = compute_regression_metrics(np.concatenate(ps), np.concatenate(ys))
    m["loss"] = total_loss / total_n
    m["mse_loss_component"] = total_mse_loss / total_n
    m["pcc_loss_component"] = total_pcc_loss / total_n
    return m

def save_ckpt(
    path, epoch, model, optimizer, scheduler, cfg, metrics,
    mean_v3, std_v3, mean_v2, std_v2,
    candidate_mean=None, candidate_std=None, candidate_raw_std=None,
    paired_mean=None, paired_std=None
):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": cfg,
        "metrics": metrics,
        "side_feature_names": [
            f"{side}.{field}" for side in ("left", "right") for field in SIDE_FIELDS
        ],
        "v3_side_mean": mean_v3,
        "v3_side_std": std_v3,
        "v2_side_mean": mean_v2,
        "v2_side_std": std_v2,
        "extra_wer_dim": getattr(model, "extra_wer_dim", 0),
        "candidate_feature_mean": candidate_mean,
        "candidate_feature_std": candidate_std,
        "candidate_feature_raw_std": candidate_raw_std,
        "paired_feature_mean": paired_mean,
        "paired_feature_std": paired_std,
    }, path)

def train(cfg, config_path, seed_override=None, smoke=False, overwrite=False):
    exp, paths = cfg["experiment"], cfg["paths"]
    wc = cfg["branches"]["whisper"]
    mc, tc = cfg["model"], cfg["training"]
    loss_name = str(tc.get("loss", "mse")).lower()
    pcc_weight = float(tc.get("pcc_weight", 0.0))
    if loss_name not in ("mse", "mse_pcc"):
        raise ValueError("training.loss must be mse or mse_pcc")
    if pcc_weight < 0:
        raise ValueError("training.pcc_weight must be >= 0")
    if loss_name == "mse" and pcc_weight != 0.0:
        print("WARNING: pcc_weight is ignored because training.loss=mse")
    validate_branch("whisper", wc)

    if cfg["fusion"]["type"] != "concat_whisper_with_dual_asr_confidence_and_wer":
        raise ValueError("D6a requires fusion.type=concat_whisper_with_dual_asr_confidence_and_wer")

    scfg_v3 = cfg["side_features_v3"]
    scfg_v2 = cfg["side_features_v2"]
    # Extra heterogeneous ASR WER sources are fully config-driven.
    # The original large-v3 Better-Ear WER remains the mandatory baseline 1-D cue.
    mode_to_dim = {"better_ear": 1, "gap": 1, "left_right": 2, "wer3": 3}
    extra_specs = []
    for spec in cfg.get("asr_wer_features", []):
        if not bool(spec.get("enabled", True)):
            continue
        name = str(spec["name"])
        mode = str(spec.get("mode", "better_ear")).lower()
        if mode not in mode_to_dim:
            raise ValueError(f"asr_wer_features[{name}].mode must be one of: better_ear, gap, left_right, wer3")
        path = str(spec["jsonl"])
        clip_max = float(spec.get("clip_max", 1.0))
        extra_specs.append({"name": name, "mode": mode, "jsonl": path,
                            "clip_max": clip_max, "dim": mode_to_dim[mode]})
    extra_wer_dim = sum(s["dim"] for s in extra_specs)

    candidate_cfg = cfg.get("candidate_feature", {})
    candidate_enabled = bool(candidate_cfg.get("enabled", False))
    candidate_dim = 1 if candidate_enabled else 0
    candidate_field = str(candidate_cfg.get("field", "candidate_max_overall"))
    candidate_root = str(candidate_cfg.get("root", ""))
    candidate_eps = float(candidate_cfg.get("normalization_eps", 1e-6))
    if candidate_enabled:
        if candidate_field != "candidate_max_overall":
            raise ValueError(
                "This controlled D6e experiment expects "
                "candidate_feature.field=candidate_max_overall"
            )
        if not candidate_root:
            raise ValueError("candidate_feature.root is required when enabled")

    paired_cfg = cfg.get("paired_feature", {})
    paired_enabled = bool(paired_cfg.get("enabled", False))
    paired_dim = int(paired_cfg.get("dim", PAIRED_FEATURE_DIM)) if paired_enabled else 0
    if paired_enabled and paired_dim != PAIRED_FEATURE_DIM:
        raise ValueError(f"paired_feature.dim must be {PAIRED_FEATURE_DIM}")
    model_extra_dim = extra_wer_dim + candidate_dim + paired_dim

    if scfg_v3.get("source") != "whisper_large_v3_confidence":
        raise ValueError("D6a expects side_features_v3.source=whisper_large_v3_confidence")
    if scfg_v2.get("source") != "whisper_large_v2_confidence":
        raise ValueError("D6a expects side_features_v2.source=whisper_large_v2_confidence")

    side_dim_v3 = 2 * len(SIDE_FIELDS)
    side_dim_v2 = 2 * len(SIDE_FIELDS)

    if int(scfg_v3["dim"]) != side_dim_v3:
        raise ValueError(f"D6a v3 side feature dim must be {side_dim_v3}")
    if int(scfg_v2["dim"]) != side_dim_v2:
        raise ValueError(f"D6a v2 side feature dim must be {side_dim_v2}")

    fused = (
        int(wc["projection_dim"])
        + side_dim_v3
        + side_dim_v2
        + 1
        + model_extra_dim
    )
    if int(cfg["fusion"]["fused_dim"]) != fused:
        raise ValueError(f"fusion.fused_dim must be {fused}")

    seed = int(seed_override if seed_override is not None else tc["seed"])
    set_seed(seed)
    run_name = f"smoke_seed{seed}" if smoke else f"seed{seed}"
    out = prepare_output_dir(
        Path(paths["output_root"]) / f"{exp['id']}_{exp['name']}" / run_name,
        overwrite=overwrite,
    )
    shutil.copy2(config_path, out / "config.yaml")
    save_json({
        **environment_info(), "seed": seed, "smoke": smoke,
        "experiment_id": exp["id"], "experiment_name": exp["name"],
    }, out / "run_info.json")

    train_ids, dev_ids = load_split_ids(paths["split_dir"])
    if len(train_ids) != 16238 or len(dev_ids) != 4059 or set(train_ids) & set(dev_ids):
        raise RuntimeError(f"Canonical split check failed: {len(train_ids)}/{len(dev_ids)}")
    metadata = load_metadata(paths["metadata"])
    if len(metadata) != 20297:
        raise RuntimeError(f"Expected 20297 metadata records, got {len(metadata)}")
    canonical_train_records = select_records_by_ids(metadata, train_ids)
    canonical_dev_records = select_records_by_ids(metadata, dev_ids)

    conf_v3, conf_v3_text = load_confidence_jsonl_with_text(paths["confidence_v3_jsonl"])
    conf_v2 = load_confidence_jsonl(paths["confidence_v2_jsonl"])
    extra_wer_tables = []
    required = set(map(str, train_ids)) | set(map(str, dev_ids))
    for spec in extra_specs:
        print(f"\nLoading extra WER source: {spec['name']} | mode={spec['mode']} | {spec['jsonl']}")
        _, source_text = load_confidence_jsonl_with_text(spec["jsonl"])
        table = build_extra_wer_table(
            source_text, metadata, mode=spec["mode"],
            clip_max=spec["clip_max"], source_name=spec["name"]
        )
        missing = sorted(required - table.keys())
        if missing:
            raise RuntimeError(
                f"{spec['name']} WER coverage incomplete: missing={len(missing)} examples={missing[:10]}"
            )
        extra_wer_tables.append((spec, table))
        print(f"{spec['name']} WER coverage: {len(required)}/{len(required)} | dim={spec['dim']}")

    extra_wer_table = None
    if extra_wer_tables:
        extra_wer_table = {}
        for sid in required:
            extra_wer_table[sid] = np.concatenate(
                [np.asarray(tab[sid], dtype=np.float32).reshape(-1) for _, tab in extra_wer_tables]
            ).astype(np.float32)

    candidate_mean = candidate_std = candidate_raw_std = None
    candidate_near_constant = None
    if candidate_enabled:
        print(
            f"\nLoading candidate feature: {candidate_field} | root={candidate_root}"
        )
        candidate_raw = load_candidate_scalar_table(
            candidate_root, required, field=candidate_field
        )
        candidate_mean, candidate_std, candidate_raw_std, candidate_near_constant = (
            fit_scalar_normalizer(
                candidate_raw,
                canonical_train_records,
                eps=candidate_eps,
            )
        )
        candidate_norm = normalize_scalar_table(
            candidate_raw, candidate_mean, candidate_std
        )
        if extra_wer_table is None:
            extra_wer_table = {}
            for sid in required:
                extra_wer_table[sid] = candidate_norm[sid]
        else:
            for sid in required:
                extra_wer_table[sid] = np.concatenate(
                    [
                        np.asarray(extra_wer_table[sid], dtype=np.float32).reshape(-1),
                        np.asarray(candidate_norm[sid], dtype=np.float32).reshape(-1),
                    ]
                ).astype(np.float32)
        print(
            f"Candidate coverage: {len(required)}/{len(required)} | "
            f"TRAIN mean={float(candidate_mean):.6f} "
            f"std={float(candidate_raw_std):.6f} | normalized dim=1"
        )

    paired_mean = paired_std = paired_raw_std = None
    if paired_enabled:
        paired_path = Path(paired_cfg["path"])
        print(f"\nLoading processed/reference features: {paired_path}", flush=True)
        paired_required = (
            {str(r["signal"]) for r in canonical_train_records[:64] + canonical_dev_records[:32]}
            if smoke else required
        )
        paired_raw = load_paired_feature_table(paired_path, paired_required, paired_dim)
        fit_records = canonical_train_records[:64] if smoke else canonical_train_records
        train_paired = np.stack(
            [paired_raw[str(r["signal"])] for r in fit_records]
        ).astype(np.float64)
        paired_mean = train_paired.mean(axis=0).astype(np.float32)
        paired_raw_std = train_paired.std(axis=0).astype(np.float32)
        paired_std = np.where(paired_raw_std < 1e-6, 1.0, paired_raw_std).astype(np.float32)
        for sid in required:
            normalized = (
                ((paired_raw[sid] - paired_mean) / paired_std).astype(np.float32)
                if sid in paired_raw else np.zeros(paired_dim, dtype=np.float32)
            )
            if extra_wer_table is None:
                extra_wer_table = {}
            if sid in extra_wer_table:
                extra_wer_table[sid] = np.concatenate(
                    [np.asarray(extra_wer_table[sid], dtype=np.float32).reshape(-1), normalized]
                )
            else:
                extra_wer_table[sid] = normalized
        save_json(
            {"fit_split": "canonical_train_only_smoke64" if smoke else "canonical_train_only",
             "path": str(paired_path),
             "dim": paired_dim, "mean": paired_mean.tolist(),
             "std_used": paired_std.tolist(), "raw_std": paired_raw_std.tolist()},
            out / "paired_normalization.json",
        )
        print(f"Paired coverage: {len(paired_required)}/{len(paired_required)} | dim={paired_dim}", flush=True)

    print("\nV3 confidence coverage")
    verify_confidence_coverage(conf_v3, metadata, train_ids, dev_ids)
    print("\nV2 confidence coverage")
    verify_confidence_coverage(conf_v2, metadata, train_ids, dev_ids)

    wer_clip_max = float(cfg["wer"]["clip_max"])
    better_ear_wer = build_better_ear_wer_table(conf_v3_text, metadata, clip_max=wer_clip_max)
    missing_wer = sorted((set(map(str, train_ids)) | set(map(str, dev_ids))) - better_ear_wer.keys())
    if missing_wer:
        raise RuntimeError(f"Better-ear WER coverage incomplete: missing={len(missing_wer)} examples={missing_wer[:10]}")

    # Fit each source independently on CANONICAL TRAIN only.
    mean_v3, std_v3, raw_std_v3, near_constant_v3 = fit_side_normalizer(
        conf_v3,
        canonical_train_records,
        eps=float(scfg_v3.get("normalization_eps", 1e-6)),
    )
    mean_v2, std_v2, raw_std_v2, near_constant_v2 = fit_side_normalizer(
        conf_v2,
        canonical_train_records,
        eps=float(scfg_v2.get("normalization_eps", 1e-6)),
    )

    feature_names = [
        f"{side}.{field}" for side in ("left", "right") for field in SIDE_FIELDS
    ]

    save_json(
        {
            "fit_split": "canonical_train_only",
            "n_fit": len(canonical_train_records),
            "feature_names": feature_names,
            "v3": {
                "mean": mean_v3.tolist(),
                "std_used": std_v3.tolist(),
                "raw_std": raw_std_v3.tolist(),
                "near_constant": near_constant_v3.tolist(),
            },
            "v2": {
                "mean": mean_v2.tolist(),
                "std_used": std_v2.tolist(),
                "raw_std": raw_std_v2.tolist(),
                "near_constant": near_constant_v2.tolist(),
            },
        },
        out / "confidence_normalization.json",
    )

    if candidate_enabled:
        save_json(
            {
                "fit_split": "canonical_train_only",
                "n_fit": len(canonical_train_records),
                "source_root": candidate_root,
                "field": candidate_field,
                "mean": float(candidate_mean),
                "std_used": float(candidate_std),
                "raw_std": float(candidate_raw_std),
                "near_constant": bool(candidate_near_constant),
            },
            out / "candidate_normalization.json",
        )

    train_records, dev_records = canonical_train_records, canonical_dev_records
    if smoke:
        train_records, dev_records = train_records[:64], dev_records[:32]

    print("\n" + "=" * 82)
    print(f"D6a D5c - WavLM | Whisper + dual ASR confidence (v3 + v2) + BetterEarWER | seed={seed} | smoke={smoke}")
    print("Whisper L32 : mean -> 1280 -> 256")
    print("ASR v3 side : left 6 + right 6 = 12 dims; canonical-TRAIN z-score")
    print("ASR v2 side : left 6 + right 6 = 12 dims; canonical-TRAIN z-score")
    print(f"Better-ear  : min(left_WER, right_WER), clipped to {wer_clip_max:g}; 1 dim")
    if paired_enabled:
        print(f"Paired audio: processed/reference spectral differences; {paired_dim} dims")
    if extra_specs:
        for spec in extra_specs:
            print(f"Extra WER   : {spec['name']} | mode={spec['mode']} | dim={spec['dim']} | clip={spec['clip_max']:g}")
        if candidate_enabled:
            print(
                f"Candidate   : {candidate_field} | zscore canonical-TRAIN only | dim=1"
            )
        print(
            f"Fusion      : 256 + 12 + 12 + 1 + {model_extra_dim} = {fused} -> 128 -> 1"
        )
    else:
        print("Extra WER   : disabled (baseline)")
        if candidate_enabled:
            print(
                f"Candidate   : {candidate_field} | zscore canonical-TRAIN only | dim=1"
            )
            print(f"Fusion      : 256 + 12 + 12 + 1 + 1 = {fused} -> 128 -> 1")
        else:
            print("Fusion      : 256 + 12 + 12 + 1 = 281 -> 128 -> 1")
    print(f"V3 confidence: {paths['confidence_v3_jsonl']}")
    print(f"V2 confidence: {paths['confidence_v2_jsonl']}")
    print(f"Train/Dev   : {len(train_records)}/{len(dev_records)}")
    print(f"Train loss  : {loss_name}" + (f" (pcc_weight={pcc_weight:g})" if loss_name == "mse_pcc" else ""))
    print("Selection   : dev_mse (unchanged from D6a baseline)")
    print(f"Output      : {out}")
    print("-" * 82)
    print("Normalizer diagnostics (canonical TRAIN only)")
    for name, mu3, sd3, mu2, sd2 in zip(
        feature_names, mean_v3, raw_std_v3, mean_v2, raw_std_v2
    ):
        print(
            f"{name:24s} "
            f"v3(mean={mu3: .6f}, std={sd3: .6f})  "
            f"v2(mean={mu2: .6f}, std={sd2: .6f})"
        )
    print("=" * 82)

    tr = build_dataset(
        train_records,
        Path(paths["whisper_feature_root"]),
        wc,
        conf_v3,
        mean_v3,
        std_v3,
        conf_v2,
        mean_v2,
        std_v2,
        better_ear_wer,
        "Loading D6a train cache",
        extra_wer=extra_wer_table,
    )
    dv = build_dataset(
        dev_records,
        Path(paths["whisper_feature_root"]),
        wc,
        conf_v3,
        mean_v3,
        std_v3,
        conf_v2,
        mean_v2,
        std_v2,
        better_ear_wer,
        "Loading D6a dev cache",
        extra_wer=extra_wer_table,
    )

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        tr, batch_size=int(tc["train_batch_size"]), shuffle=True,
        num_workers=int(tc.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(), generator=g,
    )
    dev_loader = DataLoader(
        dv, batch_size=int(tc["dev_batch_size"]), shuffle=False,
        num_workers=int(tc.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperDualConfidenceWERRegressor(
        wc, side_dim_v3, side_dim_v2, mc, extra_wer_dim=model_extra_dim
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device={device} | trainable parameters={n_trainable:,}")

    opt = AdamW(model.parameters(), lr=float(tc["learning_rate"]),
                weight_decay=float(tc["weight_decay"]))
    sch = tc["scheduler"]
    sched = ReduceLROnPlateau(
        opt, mode="min", factor=float(sch["factor"]),
        patience=int(sch["patience"]), min_lr=float(sch["min_lr"]),
    )

    max_epochs = 2 if smoke else int(tc["max_epochs"])
    patience = int(tc["early_stopping"]["patience"])
    min_delta = float(tc["early_stopping"]["min_delta"])
    grad_clip = float(tc["gradient_clip_norm"])
    keep_last = int(cfg["checkpoint"]["keep_last"])
    best_mse, best_epoch, stale, history = float("inf"), 0, 0, []

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        tm = run_epoch(model, train_loader, device, opt, grad_clip, loss_name, pcc_weight)
        dm = run_epoch(model, dev_loader, device, loss_name="mse", pcc_weight=0.0)
        sched.step(dm["mse"])
        improved = dm["mse"] < best_mse
        significant = dm["mse"] < best_mse - min_delta
        if improved:
            best_mse, best_epoch = dm["mse"], epoch
        stale = 0 if significant else stale + 1
        row = {
            "epoch": epoch, "lr": opt.param_groups[0]["lr"],
            "train_loss": tm["loss"], "train_mse_loss_component": tm["mse_loss_component"],
            "train_pcc_loss_component": tm["pcc_loss_component"],
            "train_mse": tm["mse"], "train_rmse": tm["rmse"],
            "train_pcc": tm["pcc"], "dev_loss": dm["loss"], "dev_mse": dm["mse"],
            "dev_rmse": dm["rmse"], "dev_pcc": dm["pcc"], "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{max_epochs} | train MSE={tm['mse']:.6f} PCC={tm['pcc']:.6f} | "
            f"dev MSE={dm['mse']:.6f} RMSE={dm['rmse']:.6f} PCC={dm['pcc']:.6f} | "
            f"lr={opt.param_groups[0]['lr']:.2e}" + ("  BEST" if improved else "")
        )
        ep = out / f"checkpoint_e{epoch:03d}.pt"
        save_ckpt(
            ep, epoch, model, opt, sched, cfg, dm,
            mean_v3, std_v3, mean_v2, std_v2,
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            candidate_raw_std=candidate_raw_std,
            paired_mean=paired_mean,
            paired_std=paired_std,
        )
        if improved:
            shutil.copy2(ep, out / "best_model.pt")
        ckpts = sorted(out.glob("checkpoint_e*.pt"))
        keep = {p.resolve() for p in ckpts[-keep_last:]}
        bp = out / f"checkpoint_e{best_epoch:03d}.pt"
        if bp.exists():
            keep.add(bp.resolve())
        for p in ckpts:
            if p.resolve() not in keep:
                p.unlink()
        save_history_csv(history, out / "history.csv")
        save_json(history, out / "history.json")
        if not smoke and stale >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    best = next(r for r in history if r["epoch"] == best_epoch)
    summary = {
        "experiment_id": exp["id"], "experiment_name": exp["name"],
        "seed": seed, "smoke": smoke,
        "canonical_train_samples": 16238, "canonical_dev_samples": 4059,
        "whisper": {"layer": 32, "pooling": "temporal_mean", "pooled_dim": 1280, "projection_dim": 256},
        "side_features_v3": {
            "source": "whisper_large_v3_confidence",
            "jsonl": str(paths["confidence_v3_jsonl"]),
            "dim": side_dim_v3,
            "feature_names": feature_names,
            "normalization": "zscore_fit_on_canonical_train_only",
        },
        "side_features_v2": {
            "source": "whisper_large_v2_confidence",
            "jsonl": str(paths["confidence_v2_jsonl"]),
            "dim": side_dim_v2,
            "feature_names": feature_names,
            "normalization": "zscore_fit_on_canonical_train_only",
        },
        "wer_feature": {
            "definition": "min(left_wer,right_wer)",
            "reference": "metadata.prompt",
            "hypotheses": "left/right text from whisper_large_v3 confidence JSONL",
            "clip_max": wer_clip_max,
            "dim": 1,
        },
        "extra_asr_wer_features": extra_specs,
        "candidate_feature": (
            {
                "enabled": True,
                "root": candidate_root,
                "field": candidate_field,
                "dim": 1,
                "normalization": "zscore_fit_on_canonical_train_only",
                "mean": float(candidate_mean),
                "std_used": float(candidate_std),
                "raw_std": float(candidate_raw_std),
            }
            if candidate_enabled else {"enabled": False}
        ),
        "paired_feature": (
            {"enabled": True, "path": str(paired_cfg["path"]),
             "dim": paired_dim, "normalization": "zscore_fit_on_canonical_train_only"}
            if paired_enabled else {"enabled": False}
        ),

        "fusion": "whisper_project_then_concat_with_dual_normalized_asr_confidence_5d_wer_and_candidate_max",
        "training_objective": {"loss": loss_name, "pcc_weight": pcc_weight},
        "fused_dim": fused,
        "best_epoch": best_epoch,
        "best_dev_mse": best["dev_mse"],
        "best_dev_rmse": best["dev_rmse"],
        "best_dev_pcc": best["dev_pcc"],
        "trainable_parameters": n_trainable,
        "output_dir": str(out),
    }
    save_json(summary, out / "summary.json")

    print("\n" + "=" * 82)
    print("Training complete")
    print(f"Best epoch    : {best_epoch}")
    print(f"Best dev MSE  : {best['dev_mse']:.6f}")
    print(f"Best dev RMSE : {best['dev_rmse']:.6f}")
    print(f"Best dev PCC  : {best['dev_pcc']:.6f}")
    print(f"Output        : {out}")
    print("=" * 82)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cp = a.config.expanduser().resolve()
    train(load_config(cp), cp, a.seed, a.smoke, a.overwrite)

if __name__ == "__main__":
    main()
