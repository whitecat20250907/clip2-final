#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Granite Speech 5 support starts after the repo env's pinned transformers 5.14.1.
# Keep the newer transformers isolated under the project vendor dir so training keeps
# using the original environment.
VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "transformers_5_16_1"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract IBM Granite Speech 5 TurboCTC encoder frames into CLIP2 features.f16 cache."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default="ibm-granite/granite-speech-5.0-470m-turboctc")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already written signal ids if manifest.partial.json exists.")
    parser.add_argument("--project-only", action="store_true", help="Estimate cache size from metadata durations.")
    parser.add_argument("--frames-per-second", type=float, default=None, help="Override estimate FPS for --project-only.")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state layer to cache; default last layer.")
    return parser.parse_args()


def metadata_path(data_root: Path, split: str) -> Path:
    return data_root / "metadata" / f"{split}_metadata.json"


def audio_path(data_root: Path, split: str, signal_id: str) -> Path:
    candidates = [
        data_root / split / "signals" / f"{signal_id}.flac",
        data_root / "audio" / split / "signals" / f"{signal_id}.flac",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def estimate_only(records: list[dict], data_root: Path, split: str, fps: float | None) -> None:
    durations = []
    for record in tqdm(records, desc="duration-scan"):
        ap = audio_path(data_root, split, str(record["signal"]))
        audio, _ = librosa.load(ap, sr=16000, mono=True)
        durations.append(len(audio) / 16000.0)
    total_seconds = float(np.sum(durations))
    use_fps = float(fps or 12.5)
    total_frames = int(round(total_seconds * use_fps))
    for dim in (1024,):
        gb = total_frames * dim * 2 / (1024 ** 3)
        print(json.dumps({
            "fps": use_fps,
            "dim": dim,
            "estimated_frames": total_frames,
            "estimated_cache_gb": round(gb, 3),
        }, ensure_ascii=False), flush=True)


def prepare_inputs(processor, audio: np.ndarray, device: torch.device, torch_dtype: torch.dtype) -> dict:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    cast_inputs = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if key == "input_features" and torch_dtype in (torch.float16, torch.bfloat16):
                value = value.to(dtype=torch_dtype)
        cast_inputs[key] = value
    return cast_inputs


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(metadata_path(data_root, args.split).read_text(encoding="utf-8"))
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if args.project_only:
        estimate_only(records, data_root, args.split, args.frames_per_second)
        return

    manifest_path = out_dir / "manifest.partial.json"
    index_path = out_dir / "index.json"
    final_features = out_dir / "features.f16"
    partial_features = out_dir / "features.f16.partial"
    if final_features.exists() or index_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing finalized cache in {out_dir}")

    done: list[str] = []
    offsets: list[int] = []
    lengths: list[int] = []
    total_tokens = 0
    dim = None
    if args.resume and manifest_path.exists() and partial_features.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        done = list(manifest["signal_ids"])
        offsets = list(manifest["offsets"])
        lengths = list(manifest["lengths"])
        total_tokens = int(manifest["total_tokens"])
        dim = int(manifest["dim"])
        mode = "ab"
    else:
        mode = "wb"

    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCTC.from_pretrained(args.model_id, dtype=torch_dtype, trust_remote_code=True).to(device)
    model.eval()

    start_index = len(done)
    if start_index:
        expected = [str(r["signal"]) for r in records[:start_index]]
        if expected != done:
            raise RuntimeError("Existing partial manifest does not match requested metadata order.")

    with partial_features.open(mode) as feature_file:
        for record in tqdm(records[start_index:], initial=start_index, total=len(records), desc=f"granite-{args.split}"):
            signal_id = str(record["signal"])
            audio, _ = librosa.load(audio_path(data_root, args.split, signal_id), sr=16000, mono=True)
            inputs = prepare_inputs(processor, audio, device, torch_dtype)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            if outputs.hidden_states is None:
                raise RuntimeError("Model did not return hidden_states; pass output_hidden_states=True failed.")
            hidden = outputs.hidden_states[args.layer][0]
            hidden_np = hidden.detach().to("cpu", dtype=torch.float16).numpy()
            if dim is None:
                dim = int(hidden_np.shape[1])
                print(json.dumps({
                    "detected_dim": dim,
                    "first_signal": signal_id,
                    "first_frames": int(hidden_np.shape[0]),
                    "transformers_vendor": str(VENDOR) if VENDOR.is_dir() else None,
                }, ensure_ascii=False), flush=True)
            elif int(hidden_np.shape[1]) != dim:
                raise RuntimeError(f"Feature dim changed: {hidden_np.shape[1]} != {dim}")
            if hidden_np.shape[0] <= 0 or not np.isfinite(hidden_np).all():
                raise RuntimeError(f"Bad feature for {signal_id}: shape={hidden_np.shape}")
            feature_file.write(np.ascontiguousarray(hidden_np).tobytes())
            done.append(signal_id)
            offsets.append(total_tokens)
            lengths.append(int(hidden_np.shape[0]))
            total_tokens += int(hidden_np.shape[0])
            if len(done) % 100 == 0 or len(done) == len(records):
                feature_file.flush()
                os.fsync(feature_file.fileno())
                atomic_write_json(manifest_path, {
                    "status": "partial",
                    "model": args.model_id,
                    "features": f"hidden_states[{args.layer}]",
                    "dtype": "float16",
                    "split": args.split,
                    "dim": dim,
                    "n": len(done),
                    "total_tokens": total_tokens,
                    "signal_ids": done,
                    "offsets": offsets,
                    "lengths": lengths,
                })

    if dim is None:
        raise RuntimeError("No features extracted.")
    expected_bytes = total_tokens * dim * 2
    actual_bytes = partial_features.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"features size mismatch: {actual_bytes} != {expected_bytes}")
    index = {
        "model": args.model_id,
        "features": f"hidden_states[{args.layer}]",
        "sample_rate": 16000,
        "dtype": "float16",
        "dim": dim,
        "split": args.split,
        "n": len(done),
        "total_tokens": total_tokens,
        "signal_ids": done,
        "offsets": offsets,
        "lengths": lengths,
    }
    os.replace(partial_features, final_features)
    atomic_write_json(index_path, index)
    atomic_write_json(out_dir / "extraction_summary.json", {
        "status": "complete",
        "index": str(index_path),
        "features": str(final_features),
        "n": len(done),
        "dim": dim,
        "total_tokens": total_tokens,
        "features_gb": round(final_features.stat().st_size / (1024 ** 3), 3),
        "mean_frames": float(np.mean(lengths)),
        "max_frames": int(np.max(lengths)),
    })
    if manifest_path.exists():
        manifest_path.unlink()
    print("COMPLETE " + json.dumps(json.loads((out_dir / "extraction_summary.json").read_text()), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
