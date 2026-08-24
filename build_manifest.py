#!/usr/bin/env python3
"""
Build the final clip manifest from cropped KMSAV output.

Scans kmsav_pilot/cropped/<video_id>/utts/ for *.mp4 clips, measures duration
via ffprobe, filters to the 2-10s range (per issue acceptance criteria), and
writes a manifest CSV mapping each kept clip back to its source video metadata.
"""
import csv
import subprocess
import json
from pathlib import Path

CROPPED_DIR = Path("kmsav_pilot/cropped")
CANDIDATES_CSV = Path("data/kmsav_pilot_candidates.csv")
OUTPUT_CSV = Path("data/kmsav_pilot_manifest.csv")
MIN_SEC, MAX_SEC = 2.0, 10.0


def get_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def main():
    src_meta = {}
    with open(CANDIDATES_CSV) as f:
        for row in csv.DictReader(f):
            src_meta[row["id"]] = row

    if not CROPPED_DIR.exists():
        print(f"ERROR: {CROPPED_DIR} not found. Run run_pilot_pipeline.sh first.")
        return

    rows = []
    skipped_missing_meta = 0
    skipped_out_of_range = 0
    skipped_ffprobe_fail = 0

    for video_dir in sorted(CROPPED_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        utts_dir = video_dir / "utts"
        if not utts_dir.exists():
            continue

        meta = src_meta.get(video_id)
        if meta is None:
            skipped_missing_meta += 1
            continue

        for clip_path in sorted(utts_dir.glob("*.mp4")):
            if clip_path.stem.endswith("_audio"):
                continue

            duration = get_duration(clip_path)
            if duration is None:
                skipped_ffprobe_fail += 1
                continue
            if not (MIN_SEC <= duration <= MAX_SEC):
                skipped_out_of_range += 1
                continue

            utt_id = clip_path.stem
            txt_path = utts_dir / f"{utt_id}.txt"
            transcript = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""

            rows.append({
                "clip_id": f"{video_id}_{utt_id}",
                "source_video_id": video_id,
                "domain": meta["domain"],
                "split": meta["split"],
                "duration_sec": round(duration, 2),
                "clip_path": str(clip_path),
                "transcript": transcript,
                "dataset": "KMSAV (interim, not OLKAVS)",
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "source_video_id", "domain", "split",
            "duration_sec", "clip_path", "transcript", "dataset"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written: {OUTPUT_CSV}")
    print(f"Clips kept (2-10s): {len(rows)}")
    print(f"Skipped - out of duration range: {skipped_out_of_range}")
    print(f"Skipped - ffprobe failed: {skipped_ffprobe_fail}")
    print(f"Source videos with no candidate metadata match: {skipped_missing_meta}")

    if len(rows) < 500:
        print(f"\nNOTE: only {len(rows)} clips kept, target was ~500.")

if __name__ == "__main__":
    main()