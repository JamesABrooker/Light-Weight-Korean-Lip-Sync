#!/bin/bash
# KMSAV pilot pipeline: run locally (needs youtube.com + ffmpeg + yt-dlp access)
# Candidate list: data/kmsav_pilot_candidates.csv (35 source videos, train split, <=3 participants)
set -e

RAW_DIR=./kmsav_pilot/raw
CROP_DIR=./kmsav_pilot/cropped
ASD_DIR=./kmsav_asd_v0.2
mkdir -p "$RAW_DIR" "$CROP_DIR"

# 1. Download ASD info (once)
if [ ! -d "$ASD_DIR" ]; then
  curl -L -o kmsav_asd_v0.2.zip \
    https://github.com/etri/kmsav/releases/download/v0.2.0/kmsav_asd_v0.2.zip
  unzip kmsav_asd_v0.2.zip
fi

# 2. Download + extract audio + crop for each candidate video
# IMPORTANT: read from fd 3, not stdin (fd 0) - ffmpeg reads stdin for
# interactive commands (q to quit), which corrupts the loop's input if
# they share the same file descriptor. This is why earlier runs showed
# truncated/empty video IDs partway through.
while IFS=, read -r -u 3 id domain participants len_sec split; do
  echo "=== Processing $id ($domain, ${len_sec}s) ==="

  # Download - fallback chain instead of requiring exact 1080p
  yt-dlp --no-simulate -f 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best' \
    -o "$RAW_DIR/%(id)s.%(ext)s" \
    --merge-output-format mp4 \
    "https://youtube.com/watch?v=$id" < /dev/null \
    || { echo "SKIP $id: download failed"; continue; }

  # Extract audio (16kHz mono wav) - stdin explicitly disconnected
  ffmpeg -y -i "$RAW_DIR/$id.mp4" -qscale:a 0 -ac 1 -vn -ar 16000 \
    "$RAW_DIR/$id.wav" < /dev/null

  # Crop utterances using ASD info - stdin explicitly disconnected
  python3 ../kmsav/utils/crop_video.py \
    --asdinfo-dir "$ASD_DIR" \
    --save-root "$CROP_DIR" \
    "$RAW_DIR/$id.mp4" < /dev/null \
    || { echo "SKIP $id: crop failed (no ASD info?)"; continue; }

done 3< <(tail -n +2 data/kmsav_pilot_candidates.csv)

echo "Done. Cropped utterances are in $CROP_DIR/<video_id>/utts/"