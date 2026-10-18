"""
score_lse.py

Computes LSE-D and LSE-C (Prajwal et al., 2020) using the pretrained SyncNet
from syncnet_python (Chung & Zisserman, 2016), following Wav2Lip's evaluation:
run_pipeline.py (face detection + tracking), then run_syncnet.py, per video.
  LSE-D     = SyncNet min distance (lower is better)
  LSE-C     = SyncNet confidence   (higher is better)
  av_offset = SyncNet's estimated audio-video offset in frames (0 = in sync)
averaged over the face tracks found in each clip.

Modes:
  generated: scores data/results/<tag>/<dataset>/*.mp4
  real:      scores the original, unedited test clips (reference row)

Per-clip scores are written to results/metrics/ (committed); re-running skips
clips already scored successfully.

Usage (from wav2lip-korean/, in .venv-w2l):
    .venv-w2l/bin/python scripts/score_lse.py generated kmsav --limit 3
    .venv-w2l/bin/python scripts/score_lse.py generated kmsav
    .venv-w2l/bin/python scripts/score_lse.py real lrs3
"""
import argparse
import csv
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNC = os.path.join(ROOT, "external", "syncnet_python")
RE_DIST = re.compile(r"Min dist:\s*([-\d.]+)")
RE_CONF = re.compile(r"Confidence:\s*([-\d.]+)")
RE_OFF = re.compile(r"AV offset:\s*(-?\d+)")
FIELDS = ["name", "lse_d", "lse_c", "n_tracks", "av_offset", "status", "error"]


def generated_videos(tag, dataset):
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "results", tag, dataset, "*.mp4"))):
        yield os.path.splitext(os.path.basename(path))[0], path


def real_videos(dataset, tmp):
    with open(os.path.join(ROOT, "filelists", f"eval_pairs_{dataset}.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if dataset == "lrs3":
            yield r["name"], os.path.join(ROOT, r["video"])
        else:  # KMSAV mp4s are silent: mux in their own wav
            base = os.path.splitext(os.path.join(ROOT, r["video"]))[0]
            out = os.path.join(tmp, f"{r['name']}.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", base + ".mp4", "-i", base + ".wav",
                            "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
            yield r["name"], out


def score_one(video, work):
    data_dir = os.path.join(work, "syncnet")
    common = ["--videofile", video, "--reference=clip", "--data_dir", data_dir]
    try:
        p1 = subprocess.run([sys.executable, "run_pipeline.py", *common, "--min_track", "20"],
                            cwd=SYNC, capture_output=True, text=True)
        if p1.returncode != 0:
            raise RuntimeError("pipeline: " + " ".join((p1.stderr or p1.stdout).strip().splitlines()[-1:]))
        p2 = subprocess.run([sys.executable, "run_syncnet.py", *common],
                            cwd=SYNC, capture_output=True, text=True)
        out = p2.stdout + p2.stderr
        dists = [float(x) for x in RE_DIST.findall(out)]
        confs = [float(x) for x in RE_CONF.findall(out)]
        offs = [int(x) for x in RE_OFF.findall(out)]
        if not dists or len(dists) != len(confs):
            raise RuntimeError("no face track scored")
        off = f"{statistics.mean(offs):.2f}" if offs else ""
        return statistics.mean(dists), statistics.mean(confs), len(dists), off
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["generated", "real"])
    p.add_argument("dataset", choices=["kmsav", "lrs3"])
    p.add_argument("--tag", default="baseline_wav2lip")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    label = args.tag if args.mode == "generated" else "real"
    out_dir = os.path.join(ROOT, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"lse_{label}_{args.dataset}.csv")

    done = {}
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            done = {r["name"]: r for r in csv.DictReader(f) if r["status"] == "ok"}

    rows = list(done.values())
    with tempfile.TemporaryDirectory() as tmp:
        videos = (generated_videos(args.tag, args.dataset) if args.mode == "generated"
                  else real_videos(args.dataset, tmp))
        n = 0
        for name, path in videos:
            if args.limit and n >= args.limit:
                break
            n += 1
            if name in done:
                continue
            try:
                d, c, k, off = score_one(path, tmp)
                row = {"name": name, "lse_d": f"{d:.4f}", "lse_c": f"{c:.4f}",
                       "n_tracks": k, "av_offset": off, "status": "ok", "error": ""}
            except Exception as e:
                row = {"name": name, "lse_d": "", "lse_c": "", "n_tracks": 0,
                       "av_offset": "", "status": "failed", "error": str(e)[:200]}
                print(f"[FAIL] {name}: {e}", file=sys.stderr)
            rows.append(row)
            write_csv(out_csv, rows)
            print(f"{n} {name} {row['status']} D={row['lse_d']} C={row['lse_c']} off={row['av_offset']}")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        ds = [float(r["lse_d"]) for r in ok]
        cs = [float(r["lse_c"]) for r in ok]
        sd = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0
        print(f"\n{label} / {args.dataset}: {len(ok)} scored, {len(rows) - len(ok)} failed")
        print(f"LSE-D = {statistics.mean(ds):.3f} ± {sd(ds):.3f}   (lower is better)")
        print(f"LSE-C = {statistics.mean(cs):.3f} ± {sd(cs):.3f}   (higher is better)")
        offs = [float(r["av_offset"]) for r in ok if r.get("av_offset")]
        if offs:
            print(f"AV offset = {statistics.mean(offs):.2f} ± {sd(offs):.2f} frames (0 = in sync)")
    print(f"Per-clip scores -> {out_csv}")


if __name__ == "__main__":
    main()