"""
build_wav2lip_dataset.py

Builds Wav2Lip's expected training layout from the reviewed KMSAV manifest:

    <out-root>/<video_id>/<utt_id>/0.jpg, 1.jpg, ..., audio.wav
    <filelist-dir>/train.txt, val.txt, test.txt   ("<video_id>/<utt_id>" per line)

Per clip: extracts frames as 0-indexed jpgs, copies the paired 16 kHz mono wav
to audio.wav, and checks audio/video alignment (frame count vs. audio duration
at 25 fps). Every clip gets a row in a CSV log in logs/ (status + reason).
Splits are speaker-disjoint (grouped by video_id) and seeded, since the
manifest's split column marks every clip as 'train'.

Usage (from wav2lip-korean/):
    python scripts/build_wav2lip_dataset.py data/<manifest>.csv --limit 10   # test batch
    python scripts/build_wav2lip_dataset.py data/<manifest>.csv              # full run
"""
import argparse
import csv
import datetime
import os
import random
import shutil
import subprocess
import sys

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25
SR = 16000
MAX_DRIFT_FRAMES = 2   # allowed |frames - audio_sec * fps|
MIN_FRAMES = 15        # room for Wav2Lip's 5-frame windows


def probe_fps(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    num, den = out.stdout.strip().split("/")
    return float(num) / float(den)


def build_clip(mp4, wav, out_dir):
    """Returns (n_frames, audio_sec). Raises ValueError(reason) on failure."""
    if not os.path.isfile(mp4):
        raise ValueError("missing_mp4")
    if not os.path.isfile(wav):
        raise ValueError("missing_wav")
    fps = probe_fps(mp4)
    if abs(fps - FPS) > 0.01:
        raise ValueError(f"fps_{fps:.2f}")
    info = sf.info(wav)
    if info.samplerate != SR or info.channels != 1:
        raise ValueError(f"audio_{info.samplerate}Hz_{info.channels}ch")
    audio_sec = info.frames / info.samplerate

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", mp4, "-start_number", "0",
         "-q:v", "2", os.path.join(out_dir, "%d.jpg")],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise ValueError(f"ffmpeg: {res.stderr.strip()[:200]}")

    n_frames = sum(1 for f in os.listdir(out_dir) if f.endswith(".jpg"))
    if n_frames < MIN_FRAMES:
        raise ValueError(f"too_short_{n_frames}f")
    drift = n_frames - audio_sec * FPS
    if abs(drift) > MAX_DRIFT_FRAMES:
        raise ValueError(f"misaligned_{drift:+.1f}f")

    shutil.copyfile(wav, os.path.join(out_dir, "audio.wav"))
    return n_frames, audio_sec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--clip-root", default=ROOT)
    p.add_argument("--out-root", default=os.path.join(ROOT, "data", "kmsav_wav2lip"))
    p.add_argument("--filelist-dir", default=os.path.join(ROOT, "filelists", "kmsav"))
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not os.path.isabs(args.manifest) and not os.path.exists(args.manifest):
        args.manifest = os.path.join(ROOT, args.manifest)

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"build_kmsav_{stamp}.csv")

    ok, n_fail = [], 0
    with open(log_path, "w", newline="", encoding="utf-8") as lf:
        w = csv.writer(lf)
        w.writerow(["clip_path", "video_id", "utt_id", "status",
                    "n_frames", "audio_sec", "reason"])
        for i, row in enumerate(rows, 1):
            rel = row["clip_path"]
            parts = rel.replace("\\", "/").split("/")   # .../<video_id>/utts/<utt>.mp4
            vid, utt = parts[-3], os.path.splitext(parts[-1])[0]
            mp4 = os.path.join(args.clip_root, rel)
            wav = os.path.splitext(mp4)[0] + ".wav"
            out_dir = os.path.join(args.out_root, vid, utt)
            try:
                n, sec = build_clip(mp4, wav, out_dir)
                w.writerow([rel, vid, utt, "ok", n, f"{sec:.3f}", ""])
                ok.append((vid, utt))
            except Exception as e:
                shutil.rmtree(out_dir, ignore_errors=True)
                w.writerow([rel, vid, utt, "failed", "", "", str(e)])
                print(f"[FAIL] {rel}: {e}", file=sys.stderr)
                n_fail += 1
            if i % 50 == 0:
                print(f"{i}/{len(rows)}")

    # speaker-disjoint split by video_id, targeting clip fractions
    from collections import Counter
    counts = Counter(v for v, _ in ok)
    vids = sorted(counts)
    random.Random(args.seed).shuffle(vids)

    total = len(ok)
    split_of, n_test, n_val = {}, 0, 0
    for v in vids:
        if n_test < args.test_frac * total:
            split_of[v] = "test";
            n_test += counts[v]
        elif n_val < args.val_frac * total:
            split_of[v] = "val";
            n_val += counts[v]
        else:
            split_of[v] = "train"

    lists = {"train": [], "val": [], "test": []}
    for v, u in ok:
        lists[split_of.get(v, "train")].append(f"{v}/{u}")
    os.makedirs(args.filelist_dir, exist_ok=True)
    for name, items in lists.items():
        with open(os.path.join(args.filelist_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(items) + "\n")

    print(f"\nDone. {len(ok)} ok, {n_fail} failed, {len(rows)} total.")
    for name, items in lists.items():
        n_v = len({x.split('/')[0] for x in items})
        print(f"  {name}: {len(items)} clips from {n_v} videos")
    print(f"Log -> {log_path}")
    print(f"Filelists -> {args.filelist_dir}")


if __name__ == "__main__":
    main()