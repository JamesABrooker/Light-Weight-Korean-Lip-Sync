#!/bin/bash
set -e

RAW_DIR=./kmsav_pilot/raw
CROP_DIR=./kmsav_pilot/cropped
ASD_DIR=./kmsav_asd_v0.2
mkdir -p "$RAW_DIR" "$CROP_DIR"

# Clean out old cropped failure clips
rm -rf "$CROP_DIR"/*

if [ ! -d "$ASD_DIR" ]; then
  curl -L -o kmsav_asd_v0.2.zip \
    https://github.com/etri/kmsav/releases/download/v0.2.0/kmsav_asd_v0.2.zip
  unzip kmsav_asd_v0.2.zip
fi

while IFS=, read -r -u 3 id domain participants len_sec split; do
  echo "=== Processing $id ($domain, ${len_sec}s) ==="

  # 1. Download ONLY if raw video doesn't exist yet
  if [ ! -f "$RAW_DIR/$id.mp4" ]; then
    yt-dlp --no-simulate -f 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best' \
      -o "$RAW_DIR/%(id)s.%(ext)s" \
      --merge-output-format mp4 \
      "https://youtube.com/watch?v=$id" < /dev/null \
      || { echo "SKIP $id: download failed"; continue; }
  fi

  # 2. Rescale raw video to 1920x1080 matching KMSAV bounding box dimensions
  if [ ! -f "$RAW_DIR/${id}_1080p.mp4" ]; then
    echo "Rescaling $id to 1080p frame size..."
    ffmpeg -y -i "$RAW_DIR/$id.mp4" -vf "scale=1920:1080" -c:a copy "$RAW_DIR/${id}_rescaled.mp4" < /dev/null
    mv "$RAW_DIR/${id}_rescaled.mp4" "$RAW_DIR/$id.mp4"
  fi

  # Extract audio
  ffmpeg -y -i "$RAW_DIR/$id.mp4" -qscale:a 0 -ac 1 -vn -ar 16000 \
    "$RAW_DIR/$id.wav" < /dev/null

  # 3. Crop utterances using the newly scaled 1080p raw video
  python3 ../kmsav/utils/crop_video.py \
    --asdinfo-dir "$ASD_DIR" \
    --save-root "$CROP_DIR" \
    "$RAW_DIR/$id.mp4" < /dev/null \
    || { echo "SKIP $id: crop failed (no ASD info?)"; continue; }

done 3< <(tail -n +2 data/kmsav_pilot_candidates.csv)

echo "Done. Cropped utterances regenerated in $CROP_DIR/<video_id>/utts/"