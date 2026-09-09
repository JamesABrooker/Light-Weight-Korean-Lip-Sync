#!/usr/bin/env python3
"""
Review clips in kmsav_pilot_manifest.csv: play each mp4 (video+audio),
keep/discard via keypress, write a filtered manifest.

Run from anywhere; paths are resolved relative to the repo root
(one level up from this script's `scripts/` folder).
"""

import csv
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "kmsav_pilot_manifest.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "kmsav_pilot_manifest_reviewed.csv"
PATH_COLUMN = "clip_path"  # relative to REPO_ROOT, e.g. kmsav_pilot/cropped/.../000233.mp4


def play(path: Path):
    subprocess.run(
        ["ffplay", "-autoexit", "-loglevel", "quiet", str(path)],
        check=False,
    )


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Resume support: if a reviewed file already exists, skip clips already decided.
    already_seen = set()
    kept = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            prev = list(csv.DictReader(f))
            kept = prev
            already_seen = {r["clip_id"] for r in prev if "clip_id" in r}

    total = len(rows)

    def save():
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

    for i, row in enumerate(rows, 1):
        clip_id = row.get("clip_id", "")
        if clip_id and clip_id in already_seen:
            continue

        clip_path = row[PATH_COLUMN]
        transcript = row.get("transcript", "")
        local_file = REPO_ROOT / clip_path

        print(f"\n[{i}/{total}] {clip_path}")
        print(f"  transcript: {transcript}")

        if not local_file.exists():
            print(f"  [missing file, skipping: {local_file}]")
            continue

        play(local_file)

        while True:
            choice = input("  keep (k) / discard (d) / replay (r) / quit (q): ").strip().lower()
            if choice == "r":
                play(local_file)
                continue
            break

        if choice == "q":
            print("Stopping early, progress saved.")
            save()
            return
        elif choice == "k":
            kept.append(row)
        # anything else ('d', empty, etc.) -> discard

        save()  # autosave after every decision

    save()
    print(f"\nDone. Kept {len(kept)}/{total} clips -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()