"""
audio_pipeline.py

Extracts 16kHz mono audio from video/audio files and generates 80-channel
mel-spectrograms using the exact STFT/mel recipe Wav2Lip's SyncNet expert
was trained on. This is a faithful reimplementation of the hyperparameters
and processing steps from Rudrabha/Wav2Lip's audio.py + hparams.py
(https://github.com/Rudrabha/Wav2Lip), so that LSE-D / LSE-C scores stay
comparable to the original paper's numbers.

Usage:
    # Single file
    python audio_pipeline.py extract path/to/clip.mp4 --out-dir out/

    # Batch, from a directory of clips
    python audio_pipeline.py batch path/to/clips_dir/ --out-dir out/

    # Spot-check: plot a mel next to a reference mel (e.g. a golden mel from Wav2Lip’s audio.py)
    python audio_pipeline.py spotcheck out/clip.npy --reference ref_mel.npy

    # Every accepted clip in a reviewed manifest CSV (needs a clip_path column;
    # optional clip_id column is used for output names)
    python scripts/audio_preprocessing.py manifest path/to/manifest.csv --out-dir out/ \
        [--clip-root path/to/clips/]

    # Spot-check: plot a mel next to a reference mel
    # (e.g. a golden mel from Wav2Lip's audio.py)
    python scripts/audio_preprocessing.py spotcheck out/mels/clip.npy \
        --reference tests/fixtures/english_sample_mel.npy

Outputs: <out-dir>/wav/<name>.wav and <out-dir>/mels/<name>.npy (shape 80 x T).
batch and manifest exit non-zero if any file fails.

Requires: librosa, soundfile, numpy, scipy, pandas, matplotlib (for spotcheck only)
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
import subprocess

# Hyperparameters — must exactly match Wav2Lip's hparams.py.
# Do not change these without also retraining/re-evaluating SyncNet.
HPARAMS = dict(
    sample_rate=16000,
    n_fft=800,
    hop_size=200,
    win_size=800,
    num_mels=80,
    fmin=55,
    fmax=7600,
    min_level_db=-100,
    ref_level_db=20,
    preemphasis_val=0.97,
    preemphasize=True,
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.0,
)


# audio extraction / resampling
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".m4a")


def load_and_resample(path, sr=HPARAMS["sample_rate"]):
    """Load any audio/video file, downmix to mono, resample to 16kHz.
    Video containers are decoded with ffmpeg (libsndfile can't read them)."""
    if path.lower().endswith(VIDEO_EXTS):
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn",
               "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0 or not res.stdout:
            raise RuntimeError(f"ffmpeg failed on {path}: {res.stderr.decode().strip()}")
        return np.frombuffer(res.stdout, dtype=np.float32).copy()
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def save_wav(y, out_path, sr=HPARAMS["sample_rate"]):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, y, sr)


# mel-spectrogram generation (Wav2Lip's exact recipe)
def _preemphasis(wav, k, preemphasize):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def _stft(y, n_fft, hop_size, win_size):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size)


_mel_basis_cache = {}


def _build_mel_basis(sr, n_fft, num_mels, fmin, fmax):
    key = (sr, n_fft, num_mels, fmin, fmax)
    if key not in _mel_basis_cache:
        _mel_basis_cache[key] = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
    return _mel_basis_cache[key]


def _linear_to_mel(spectrogram, mel_basis):
    return np.dot(mel_basis, spectrogram)


def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S, hp):
    m, mn = hp["max_abs_value"], hp["min_level_db"]
    if hp["symmetric_mels"]:
        return np.clip((2 * m) * ((S - mn) / -mn) - m, -m, m)
    return np.clip(m * ((S - mn) / -mn), 0, m)


def melspectrogram(wav, hp=HPARAMS):
    """Returns an (80, T) mel-spectrogram matching Wav2Lip's SyncNet training config."""
    D = _stft(
        _preemphasis(wav, hp["preemphasis_val"], hp["preemphasize"]),
        hp["n_fft"],
        hp["hop_size"],
        hp["win_size"],
    )
    mel_basis = _build_mel_basis(
        hp["sample_rate"], hp["n_fft"], hp["num_mels"], hp["fmin"], hp["fmax"]
    )
    S = _amp_to_db(_linear_to_mel(np.abs(D), mel_basis), hp["min_level_db"]) - hp["ref_level_db"]
    return _normalize(S, hp)


# verification against expected config
def verify_hparams(hp=HPARAMS):
    """Assert hparams exactly match SyncNet's original training config."""
    expected = dict(
        sample_rate=16000, n_fft=800, hop_size=200, win_size=800,
        num_mels=80, fmin=55, fmax=7600,
        min_level_db=-100, ref_level_db=20,
        preemphasis_val=0.97, preemphasize=True,
        symmetric_mels=True, max_abs_value=4.0,
    )
    mismatches = {k: (hp[k], v) for k, v in expected.items() if hp[k] != v}
    if mismatches:
        raise ValueError(f"hparams mismatch vs SyncNet training config: {mismatches}")
    print("hparams verified OK against SyncNet training config:")
    print(json.dumps(expected, indent=2))


def check_mel_shape(mel, wav_len, hp=HPARAMS):
    """Sanity check mel shape against expected frame count for the given wav length."""
    expected_frames = wav_len // hp["hop_size"]
    actual_frames = mel.shape[1]
    ok = abs(expected_frames - actual_frames) <= 2  # librosa framing/padding slack
    status = "OK" if ok else "MISMATCH"
    print(f"mel shape: {mel.shape} (expected ~{hp['num_mels']} x {expected_frames}) -> {status}")
    if not ok:
        raise ValueError(f"mel frame count {actual_frames} != expected ~{expected_frames}")
    return ok


# Pipeline entry points
def process_file(in_path, out_dir, name=None):
    name = name or os.path.splitext(os.path.basename(in_path))[0]
    wav = load_and_resample(in_path)
    save_wav(wav, os.path.join(out_dir, "wav", f"{name}.wav"))

    mel = melspectrogram(wav)
    check_mel_shape(mel, len(wav))

    mel_dir = os.path.join(out_dir, "mels")
    os.makedirs(mel_dir, exist_ok=True)
    mel_path = os.path.join(mel_dir, f"{name}.npy")
    np.save(mel_path, mel)
    print(f"[{name}] wav={len(wav)/HPARAMS['sample_rate']:.2f}s  mel={mel.shape}  -> {mel_path}")
    return mel_path


def process_batch(in_dir, out_dir):
    verify_hparams()
    exts = (".mp4", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".m4a")
    files = [f for f in sorted(os.listdir(in_dir)) if f.lower().endswith(exts)]
    if not files:
        print(f"No audio/video files found in {in_dir}")
        return
    n_ok, n_fail = 0, 0
    for f in files:
        try:
            process_file(os.path.join(in_dir, f), out_dir)
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] {f}: {e}", file=sys.stderr)
            n_fail += 1
    print(f"Done. {n_ok} succeeded, {n_fail} failed, {len(files)} total.")
    if n_fail:
        sys.exit(1)


def process_manifest(manifest_path, out_dir, clip_root=None):
    """Process every row in a reviewed manifest CSV (already-accepted clips only)."""
    verify_hparams()

    df = pd.read_csv(manifest_path)
    if "clip_path" not in df.columns:
        raise ValueError(
            f"'clip_path' column not found in {manifest_path}. "
            f"Columns present: {list(df.columns)}"
        )

    base_dir = clip_root or os.path.dirname(os.path.abspath(manifest_path))
    has_id = "clip_id" in df.columns
    print(f"Processing {len(df)} accepted clips from {manifest_path}")
    print(f"Resolving clip_path against: {base_dir}")

    n_ok, n_fail = 0, 0
    for _, row in df.iterrows():
        rel_path = os.path.splitext(row["clip_path"])[0] + ".wav"
        full_path = os.path.join(base_dir, rel_path)
        name = str(row["clip_id"]) if has_id else os.path.splitext(os.path.basename(rel_path))[0]
        try:
            process_file(full_path, out_dir, name=name)
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] {rel_path}: {e}", file=sys.stderr)
            n_fail += 1

    print(f"Done. {n_ok} succeeded, {n_fail} failed, {len(df)} total.")


def spotcheck(mel_path, reference_path=None, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = np.load(mel_path)
    n_plots = 2 if reference_path else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    axes = np.atleast_1d(axes)

    axes[0].imshow(mel, aspect="auto", origin="lower")
    axes[0].set_title(f"Generated mel: {os.path.basename(mel_path)} shape={mel.shape}")

    if reference_path:
        ref = np.load(reference_path)
        axes[1].imshow(ref, aspect="auto", origin="lower")
        axes[1].set_title(f"Reference mel: {os.path.basename(reference_path)} shape={ref.shape}")
        print(f"generated range=[{mel.min():.3f}, {mel.max():.3f}]  "
              f"reference range=[{ref.min():.3f}, {ref.max():.3f}]")

    out_png = out_png or (os.path.splitext(mel_path)[0] + "_spotcheck.png")
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved spot-check plot -> {out_png}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Process a single file")
    p_extract.add_argument("input")
    p_extract.add_argument("--out-dir", default="out")

    p_batch = sub.add_parser("batch", help="Process all files in a directory")
    p_batch.add_argument("input_dir")
    p_batch.add_argument("--out-dir", default="out")

    p_manifest = sub.add_parser("manifest", help="Process all clips listed in a reviewed manifest CSV")
    p_manifest.add_argument("manifest_path")
    p_manifest.add_argument("--out-dir", default="out")
    p_manifest.add_argument("--clip-root", default=None,
                             help="Base dir to resolve clip_path against (default: manifest's own dir)")

    p_verify = sub.add_parser("verify", help="Verify hparams against SyncNet config")

    p_spot = sub.add_parser("spotcheck", help="Plot a mel (optionally vs. a reference mel)")
    p_spot.add_argument("mel_path")
    p_spot.add_argument("--reference", default=None)
    p_spot.add_argument("--out-png", default=None)

    args = parser.parse_args()

    if args.cmd == "extract":
        verify_hparams()
        process_file(args.input, args.out_dir)
    elif args.cmd == "batch":
        process_batch(args.input_dir, args.out_dir)
    elif args.cmd == "manifest":
        process_manifest(args.manifest_path, args.out_dir, args.clip_root)
    elif args.cmd == "verify":
        verify_hparams()
    elif args.cmd == "spotcheck":
        spotcheck(args.mel_path, args.reference, args.out_png)


if __name__ == "__main__":
    main()