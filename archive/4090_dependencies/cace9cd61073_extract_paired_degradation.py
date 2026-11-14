"""Extract alignment-free processed/reference acoustic differences for CLIP2.

The reference is the matching unprocessed mixture, not a vocal stem.  These
features describe changes to the same excerpt and use no correctness labels.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import soxr
except ImportError:
    soxr = None


SAMPLE_RATE = 16_000
N_FFT = 512
HOP = 160
BANDS = ((150, 500), (500, 1500), (1500, 3500), (3500, 7600))
FEATURE_NAMES = tuple(
    name
    for ear in ("left", "right")
    for name in (
        *(f"{ear}.level_delta_db.{low}_{high}Hz" for low, high in BANDS),
        *(f"{ear}.envelope_corr.{low}_{high}Hz" for low, high in BANDS),
        *(f"{ear}.onset_corr.{low}_{high}Hz" for low, high in BANDS[2:]),
    )
)


def read_stereo(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] != 2:
        raise ValueError(f"Expected mono/stereo audio: {path} has {audio.shape[1]} channels")
    if sr != SAMPLE_RATE:
        if soxr is not None:
            audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
        else:
            import torch
            import torchaudio
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio.T.copy()), sr, SAMPLE_RATE
            ).T.numpy()
    return np.asarray(audio, dtype=np.float32)


def band_power(signal: np.ndarray) -> np.ndarray:
    if signal.size < N_FFT:
        signal = np.pad(signal, (0, N_FFT - signal.size))
    frames = np.lib.stride_tricks.sliding_window_view(signal, N_FFT)[::HOP]
    spectrum = np.fft.rfft(frames * np.hanning(N_FFT), axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    return np.stack(
        [power[:, (freqs >= low) & (freqs < high)].sum(axis=1)
         for low, high in BANDS],
        axis=0,
    )


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / denom) if denom > 1e-10 else 0.0


def shift_pair(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        a, b = a[..., lag:], b[..., :-lag]
    elif lag < 0:
        a, b = a[..., :lag], b[..., -lag:]
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def channel_features(processed: np.ndarray, reference: np.ndarray) -> list[float]:
    proc = band_power(processed)
    ref = band_power(reference)
    proc_env = np.log1p(proc)
    ref_env = np.log1p(ref)
    lag = max(
        range(-4, 5),
        key=lambda k: correlation(
            *shift_pair(proc_env.sum(axis=0), ref_env.sum(axis=0), k)
        ),
    )
    proc, ref = shift_pair(proc, ref, lag)
    p_log, r_log = np.log1p(proc), np.log1p(ref)
    level = 10 * np.log10((proc.mean(axis=1) + 1e-6) / (ref.mean(axis=1) + 1e-6))
    level = np.clip(level, -40, 40)
    envelope = [correlation(p_log[i], r_log[i]) for i in range(len(BANDS))]
    onset = [correlation(np.maximum(np.diff(p_log[i]), 0),
                         np.maximum(np.diff(r_log[i]), 0)) for i in (2, 3)]
    return [*map(float, level), *envelope, *onset]


def process_record(record: dict, root: Path, split: str) -> dict:
    sid = str(record["signal"])
    processed_path = root / split / "signals" / f"{sid}.flac"
    reference_path = root / split / "unprocessed" / f"{sid}_unproc.flac"
    processed = read_stereo(processed_path)
    reference = read_stereo(reference_path)
    values = []
    for ear in range(2):
        values.extend(channel_features(processed[:, ear], reference[:, ear]))
    if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError(f"{sid}: invalid paired features")
    return {"signal": sid, "features": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "eval"), default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("--workers must be >=1 and --limit must be >=0")
    metadata_path = args.data_root / "metadata" / f"{args.split}_metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]
    expected = {str(row["signal"]) for row in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate signal IDs in metadata")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if len(item["features"]) != len(FEATURE_NAMES):
                        raise ValueError(f"Bad feature dimension in {args.out}")
                    sid = str(item["signal"])
                    if sid in existing:
                        raise ValueError(f"Duplicate {sid} in {args.out}")
                    existing.add(sid)
    if existing - expected:
        raise ValueError("Output contains IDs outside requested split/limit")
    todo = [row for row in records if str(row["signal"]) not in existing]
    print(f"paired features: split={args.split} total={len(records)} "
          f"done={len(existing)} remaining={len(todo)} dim={len(FEATURE_NAMES)}", flush=True)
    work = partial(process_record, root=args.data_root, split=args.split)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.out.open("a", encoding="utf-8") as f:
        for item in tqdm(pool.map(work, todo), total=len(todo), unit="sample"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
    schema_path = args.out.with_suffix(".schema.json")
    schema_path.write_text(
        json.dumps({"split": args.split, "n": len(records), "features": FEATURE_NAMES,
                    "sample_rate": SAMPLE_RATE, "source": "processed versus unprocessed"}, indent=2),
        encoding="utf-8",
    )
    print(f"complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
