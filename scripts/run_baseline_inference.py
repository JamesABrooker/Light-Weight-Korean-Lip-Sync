"""
run_baseline_inference.py

Runs Wav2Lip's inference.py on every pair in filelists/eval_pairs_<dataset>.csv.
- KMSAV: clips are already face crops (verified equivalent to S3FD for the
  sync expert), so a fixed full-frame box is used.
- LRS3: S3FD face detection (inference.py default), as its face tracks are
  framed too loosely.
Outputs go to data/results/<tag>/<dataset>/<name>.mp4; a CSV log goes to logs/.
Existing outputs are skipped, so the run can be resumed.

Usage (from wav2lip-korean/, in .venv-w2l):
    .venv-w2l/bin/python scripts/run_baseline_inference.py kmsav --limit 1
    .venv-w2l/bin/python scripts/run_baseline_inference.py kmsav
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W2L = os.path.join(ROOT, "external", "Wav2Lip")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["kmsav", "lrs3"])
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "wav2lip.pth"))
    p.add_argument("--tag", default="baseline_wav2lip")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    with open(os.path.join(ROOT, "filelists", f"eval_pairs_{args.dataset}.csv")) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    out_dir = os.path.join(ROOT, "data", "results", args.tag, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(W2L, "temp"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(ROOT, "logs", f"inference_{args.tag}_{args.dataset}_{stamp}.csv")
    extra = ["--box", "0", "224", "0", "224"] if args.dataset == "kmsav" else []

    n_ok = n_fail = n_skip = 0
    with open(log_path, "w", newline="") as lf:
        w = csv.writer(lf)
        w.writerow(["name", "status", "seconds", "error"])
        for i, r in enumerate(rows, 1):
            out = os.path.join(out_dir, f"{r['name']}.mp4")
            if os.path.exists(out):
                w.writerow([r["name"], "skipped", "", ""])
                n_skip += 1
                continue
            cmd = [sys.executable, "inference.py",
                   "--checkpoint_path", args.checkpoint,
                   "--face", os.path.join(ROOT, r["video"]),
                   "--audio", os.path.join(ROOT, r["audio"]),
                   "--outfile", out] + extra
            t0 = time.time()
            res = subprocess.run(cmd, cwd=W2L, capture_output=True, text=True)
            dt = time.time() - t0
            if res.returncode == 0 and os.path.exists(out):
                w.writerow([r["name"], "ok", f"{dt:.1f}", ""])
                n_ok += 1
            else:
                err = " ".join((res.stderr or res.stdout).strip().splitlines()[-1:])
                w.writerow([r["name"], "failed", f"{dt:.1f}", err])
                print(f"[FAIL] {r['name']}: {err}", file=sys.stderr)
                n_fail += 1
            lf.flush()
            print(f"{i}/{len(rows)} {r['name']} {dt:.1f}s")

    print(f"\nDone. {n_ok} ok, {n_fail} failed, {n_skip} skipped. Log -> {log_path}")


if __name__ == "__main__":
    main()