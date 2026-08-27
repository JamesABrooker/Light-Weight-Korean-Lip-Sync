#!/bin/bash
set -e

RAW_DIR=./kmsav_pilot/raw
CROP_DIR=./kmsav_pilot/cropped
ASD_DIR=./kmsav_asd_v0.2
mkdir -p "$RAW_DIR" "$CROP_DIR"

# Clean out old cropped failure clips
rm -rf "$CROP_DIR"/*
rm -f "$RAW_DIR"/*.mp4 "$RAW_DIR"/*.wav

if [ ! -d "$ASD_DIR" ]; then
  curl -L -o kmsav_asd_v0.2.zip \
    https://github.com/etri/kmsav/releases/download/v0.2.0/kmsav_asd_v0.2.zip
  unzip kmsav_asd_v0.2.zip
fi

while IFS=, read -r -u 3 id domain participants len_sec split; do
  echo "=== Processing $id ($domain, ${len_sec}s) ==="

  # Download ONLY if raw video doesn't exist yet
  if [ ! -f "$RAW_DIR/$id.mp4" ]; then
    yt-dlp --no-simulate -f 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best' \
      -o "$RAW_DIR/%(id)s.%(ext)s" \
      --merge-output-format mp4 \
      "https://youtube.com/watch?v=$id" < /dev/null \
      || { echo "SKIP $id: download failed"; continue; }
  fi

  # Extract audio
  ffmpeg -y -i "$RAW_DIR/$id.mp4" -qscale:a 0 -ac 1 -vn -ar 16000 \
    "$RAW_DIR/$id.wav" < /dev/null

  # Determine this video's own ASD-declared frame size, then match it
  asd_txt=$(find "$ASD_DIR/$id" -iname "??????.txt" | head -n1)
  if [ -z "$asd_txt" ]; then
    echo "SKIP $id: no ASD info found"; continue
  fi
  read -r img_w img_h <<< "$(grep -m1 '# ImageSize' "$asd_txt" | cut -d: -f2)"

  echo "Rescaling $id to match ASD ImageSize ${img_w}x${img_h}..."
  ffmpeg -y -i "$RAW_DIR/$id.mp4" -vf "scale=${img_w}:${img_h}" -c:a copy \
    "$RAW_DIR/${id}_rescaled.mp4" < /dev/null
  mv "$RAW_DIR/${id}_rescaled.mp4" "$RAW_DIR/$id.mp4"

  # Crop
  python3 ../kmsav/utils/crop_video.py \
    --asdinfo-dir "$ASD_DIR" \
    --save-root "$CROP_DIR" \
    "$RAW_DIR/$id.mp4" < /dev/null \
    || { echo "SKIP $id: crop failed (no ASD info?)"; continue; }

done 3< <(tail -n +2 data/kmsav_pilot_candidates.csv)

echo "Done. Cropped utterances regenerated in $CROP_DIR/<video_id>/utts/"