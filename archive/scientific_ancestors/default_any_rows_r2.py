"""Frozen 583593 default inference from newly generated label-free feature arrays.

Input begins after the 119/58/30 backbone features and independent frame/query
preparation. No canonical VALID cache or cached predictions are opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .prediction_bank import sha256


FILE_NAMES = ("features.npz", "queries.npz", "frame_index.npz", "frames.f16")
FEATURE_KEYS = ("signal_ids", "acoustic", "side", "words", "word_mask",
                "severity", "n_words", "text")
QUERY_KEYS = ("signal_ids", "text", "phone")
INDEX_KEYS = ("signal_ids", "offsets", "spans")
SEEDS = (42, 43, 44)


def _inside(path: Path, root: Path) -> Path:
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError(f"Frozen asset leaves owned work root: {path}")
    return path


def _read_exact_npz(path: Path, expected_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_keys):
            raise ValueError(f"{path.name} keys differ from label-free inference schema")
        return {key: archive[key].copy() for key in expected_keys}


def load_inputs(folder: Path, expected_manifest_sha256: str) -> tuple[dict, dict, dict, np.memmap, dict]:
    folder = Path(folder).resolve(strict=True)
    manifest_path = folder / "manifest.json"
    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ValueError("Expected manifest SHA-256 must be a lowercase 64-char digest")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Feature manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "v8_default_119_58_30_frames64_queries32_r1" or set(
            manifest.get("files", {})) != set(FILE_NAMES):
        raise ValueError("Unexpected default feature manifest schema")
    for name in FILE_NAMES:
        digest = manifest["files"][name]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid {name} digest")
        if sha256(folder / name) != digest:
            raise ValueError(f"{name} SHA-256 mismatch")
    features = _read_exact_npz(folder / "features.npz", FEATURE_KEYS)
    queries = _read_exact_npz(folder / "queries.npz", QUERY_KEYS)
    index = _read_exact_npz(folder / "frame_index.npz", INDEX_KEYS)
    ids = features["signal_ids"]
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(ids) < 1:
        raise ValueError("signal_ids must be a nonempty string vector")
    ids = ids.astype(str)
    if len(set(ids.tolist())) != len(ids) or any(
            not sid or "," in sid or "\n" in sid or "\r" in sid for sid in ids):
        raise ValueError("Invalid or duplicate signal_ids")
    features["signal_ids"] = ids
    n = len(ids)
    if not np.array_equal(ids, queries["signal_ids"]) or not np.array_equal(
            ids, index["signal_ids"]):
        raise ValueError("Feature/query/frame IDs mismatch")
    acoustic, side, words = (features[k] for k in ("acoustic", "side", "words"))
    mask, severity, counts, text = (features[k] for k in (
        "word_mask", "severity", "n_words", "text"))
    if acoustic.shape != (n, 1280) or side.shape != (n, 30):
        raise ValueError("Default requires acoustic[n,1280] and side[n,30]")
    if words.ndim != 3 or words.shape[0] != n or words.shape[2] != 119:
        raise ValueError("Default requires words[n,max_words,119]")
    if text.shape != (n, 58) or mask.shape != words.shape[:2] or mask.dtype.kind != "b":
        raise ValueError("Default requires text[n,58] and boolean word_mask")
    if severity.shape != (n,) or severity.dtype.kind not in "iu" or not np.isin(
            severity, (0, 1, 2)).all():
        raise ValueError("severity must be integer 0/1/2")
    if counts.shape != (n,) or counts.dtype.kind not in "iu" or (counts < 1).any() or not np.array_equal(
            mask.sum(1), counts):
        raise ValueError("n_words must equal non-padded word count")
    if any(array.dtype.kind not in "fiu" or not np.isfinite(array).all()
           for array in (acoustic, side, words, text)) or np.any(words[~mask] != 0):
        raise ValueError("Backbone features must be finite numeric with zero word padding")
    qtext, phone = queries["text"], queries["phone"]
    if qtext.shape != (*mask.shape, 32) or qtext.dtype.kind != "f" or not np.isfinite(qtext).all():
        raise ValueError("Attention text must be finite [n,max_words,32]")
    if phone.shape != (*mask.shape, 32) or phone.dtype.kind not in "iu" or (phone < 0).any():
        raise ValueError("Attention phone must be nonnegative integer [n,max_words,32]")
    offsets, spans = index["offsets"], index["spans"]
    if offsets.shape != (n + 1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0 or np.any(
            np.diff(offsets) < 1) or np.any(np.diff(offsets) > 1500):
        raise ValueError("Invalid per-row frame offsets")
    if spans.shape != (*mask.shape, 2) or spans.dtype.kind not in "iu":
        raise ValueError("Invalid attention spans")
    lengths = np.diff(offsets)
    # Frozen alignment can assign a real word a zero-width span. The trained
    # soft attention path handles this through its minimum prior width.
    if np.any(spans[mask, 0] < 0) or np.any(spans[mask, 1] < spans[mask, 0]) or np.any(
            spans[..., 1][mask] > np.broadcast_to(lengths[:, None], mask.shape)[mask]):
        raise ValueError("Active word spans leave their row frames")
    frame_path = folder / "frames.f16"
    if frame_path.stat().st_size != int(offsets[-1]) * 64 * np.dtype(np.float16).itemsize:
        raise ValueError("Frame byte count differs from offsets")
    frames = np.memmap(frame_path, mode="r", dtype=np.float16, shape=(int(offsets[-1]), 64))
    if not np.isfinite(frames).all():
        raise ValueError("Nonfinite projected attention frame")
    return features, queries, index, frames, manifest


def default_csv_bytes(ids: np.ndarray, prediction: np.ndarray) -> bytes:
    if prediction.shape != (len(ids),) or not np.isfinite(prediction).all() or np.any(
            (prediction < 0) | (prediction > 1)):
        raise ValueError("Invalid default CSV prediction")
    return "".join(f"{sid},{float(score) * 100:.6f}\n"
                   for sid, score in zip(ids, prediction)).encode("utf-8")


def _checkpoint(root: Path, registry: dict, seed: int) -> tuple[Path, Path, str]:
    cp = _inside(root / registry["component_checkpoint_patterns"]["default"].format(seed=seed), root)
    report = _inside(root / "runs/v6_equal_attention_20260924/equal_stems_word_attention"
                     / f"seed{seed}/report.json", root)
    expected = registry["component_checkpoints"]["default"][str(seed)]
    if sha256(cp) != expected:
        raise ValueError(f"Default seed {seed} checkpoint SHA mismatch")
    metadata = json.loads(report.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed_seed" or metadata.get("checkpoint_sha256") != expected:
        raise ValueError(f"Default seed {seed} report identity mismatch")
    return cp, report, expected


def infer(root: Path, source_root: Path, registry_path: Path, input_folder: Path,
          expected_manifest_sha256: str, output: Path, *, device: str = "cuda:0",
          batch_size: int = 64) -> dict:
    import torch
    from d6e_word_v1 import train as v1
    from d6e_word_v1.common import original_module
    from fusion_v2.v6_attention_model_20260924 import AttentionResidual
    from fusion_v2.v6_attention_stems_train_20260924 import predict as attention_predict
    from fusion_v2.v6_m2_stem_fusion_20260924 import model_class

    root = Path(root).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    input_folder = _inside(input_folder, root)
    output = Path(output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Output must be a fresh directory inside the owned work root")
    if batch_size < 1 or device != "cuda:0":
        raise ValueError("Default frozen replay requires positive batch size and mapped cuda:0")
    registry_path = Path(registry_path).resolve(strict=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("official_submission_id") != 584197 or registry.get("seeds", {}).get(
            "default") != list(SEEDS):
        raise ValueError("Expected adopted 584197 default three-seed registry")
    features, queries, index, frames, manifest = load_inputs(input_folder, expected_manifest_sha256)
    for name in ("v6_equal_attention_20260924", "v6_attention_stems_train_20260924",
                 "v6_attention_model_20260924", "v6_m2_stem_fusion_20260924"):
        source = _inside(root / "03_my_improvements/fusion_v2" / (name + ".py"), root)
        if sha256(source) != manifest["frozen_source_sha256"][name]:
            raise ValueError(f"Frozen V6 source changed: {name}")
    original_path = source_root / "clarity/recipes/clip2/stage1_20260811/scripts/"
    original_path /= "train_d6e_multicandidate_max.py"
    if sha256(original_path) != manifest["original_source_sha256"]:
        raise ValueError("Frozen D6e model source changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    original, _ = original_module(source_root)
    extra = {**queries, **index, "frames": frames}
    ids = features["signal_ids"]
    predictions, base_members, attention_members = [], [], []
    checkpoint_shas, report_shas, component_shas = {}, {}, {}
    with torch.inference_mode():
        for seed in SEEDS:
            cp, report, expected_sha = _checkpoint(root, registry, seed)
            bundle = torch.load(cp, map_location="cpu", weights_only=False)
            if bundle.get("seed") != seed or bundle.get("recipe", {}).get("weights") != [0.5, 0.5]:
                raise ValueError(f"Default seed {seed} frozen recipe mismatch")
            if bundle["recipe"]["sources"] != manifest["frozen_source_sha256"]:
                raise ValueError(f"Default seed {seed} source recipe mismatch")
            base_cp = _inside(root / f"runs/v6_m2_stems_20260924/m2_stem_add/seed{seed}/"
                              f"m2_stem_add_seed{seed}/best_model.pt", root)
            attention_cp = _inside(root / f"runs/v6_attention_stems_20260924/"
                                   f"text_word_attention/seed{seed}/best_model.pt", root)
            if (sha256(base_cp) != bundle["component_sha256"]["base"] or
                    sha256(attention_cp) != bundle["component_sha256"]["attention"]):
                raise ValueError(f"Default seed {seed} bundled component identity mismatch")
            base_ck = bundle["base"]
            norm = base_ck["normalizer"]
            word = (features["words"] - norm["word_mean"]) / norm["word_std"]
            word = np.where(features["word_mask"][..., None], word, 0).astype(np.float32)
            text = ((features["text"] - norm["text_mean"]) / norm["text_std"]).astype(np.float32)
            arrays = (features["acoustic"].astype(np.float32),
                      features["side"].astype(np.float32), word,
                      features["word_mask"].astype(np.bool_),
                      features["severity"].astype(np.int64),
                      features["n_words"].astype(np.int64), text)
            if any(not np.isfinite(a).all() for a in arrays if a.dtype.kind == "f"):
                raise ValueError(f"Default seed {seed} normalizer gave nonfinite inputs")
            model = model_class("m2_stem_add")(
                v1.make_base(original, base_ck["base_config"]), "word", 119, 58).to(device)
            model.load_state_dict(base_ck["model_state"], strict=True)
            model.eval()
            base_parts, word_parts = [], []
            for start in range(0, len(ids), batch_size):
                batch = [torch.from_numpy(a[start:start + batch_size]).to(device) for a in arrays]
                output_batch = model(*batch)
                base_parts.append(output_batch["prediction"].cpu().numpy())
                word_parts.append(output_batch["word_probs"].cpu().numpy())
            base = np.concatenate(base_parts)
            word_probs = np.concatenate(word_parts)
            del model
            attention_ck = bundle["attention"]
            if (attention_ck.get("recipe") != bundle["recipe"] and
                    attention_ck.get("recipe", {}).get("arm") != "text_word_attention"):
                raise ValueError(f"Default seed {seed} attention recipe mismatch")
            if np.any(queries["phone"] >= attention_ck["recipe"]["phone_vocab"]):
                raise ValueError("Attention phone ID outside frozen vocabulary")
            attention = AttentionResidual(attention_ck["recipe"]["phone_vocab"],
                                          "text_word_attention").to(device)
            attention.load_state_dict(attention_ck["model_state"], strict=True)
            attention.eval()
            # The frozen attention helper consumes only inputs, base predictions and word probabilities.
            residual, _, _ = attention_predict(attention, features, extra, base,
                                               word_probs, attention_ck["norm"])
            del attention
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            seed_prediction = (base.astype(np.float64) + residual.astype(np.float64)) / 2
            if seed_prediction.shape != (len(ids),) or not np.isfinite(seed_prediction).all():
                raise ValueError(f"Default seed {seed} invalid prediction")
            predictions.append(seed_prediction)
            base_members.append(base)
            attention_members.append(residual)
            checkpoint_shas[str(seed)] = expected_sha
            report_shas[str(seed)] = sha256(report)
            component_shas[str(seed)] = bundle["component_sha256"]
    overall = np.mean(np.stack(predictions), axis=0, dtype=np.float64)
    csv_bytes = default_csv_bytes(ids, overall)
    output.mkdir(parents=True)
    prediction_path = output / "predictions.npz"
    np.savez_compressed(prediction_path, signal_ids=ids, prediction=overall,
                        seed_predictions=np.stack(predictions).astype(np.float32),
                        base_seed_predictions=np.stack(base_members).astype(np.float32),
                        attention_seed_predictions=np.stack(attention_members).astype(np.float32))
    (output / "default_predictions.csv").write_bytes(csv_bytes)
    report = {"status": "completed_frozen_default_arbitrary_rows", "rows": len(ids),
              "input_manifest_sha256": expected_manifest_sha256,
              "input_file_sha256": manifest["files"],
              "registry_sha256": sha256(registry_path),
              "adapter_source_sha256": sha256(Path(__file__)),
              "checkpoint_sha256": checkpoint_shas,
              "training_report_sha256": report_shas,
              "component_sha256": component_shas,
              "predictions_sha256": sha256(prediction_path),
              "default_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
              "prediction_scale": "fraction_0_1; CSV percent six decimals",
              "raw_audio_processed": False, "valid_cache_read": False, "labels_read": False,
              "official_submission_created": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = infer(args.work_root, args.source_root, args.registry, args.input_folder,
                   args.expected_manifest_sha256, args.output, device=args.device,
                   batch_size=args.batch_size)
    print(json.dumps({"status": result["status"], "rows": result["rows"],
                      "default_csv_sha256": result["default_csv_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
