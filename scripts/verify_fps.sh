#!/usr/bin/env bash
# verify_fps.sh - confirm frame count matches expected duration * fps
CLIP="$1"
EXPECTED_FPS=25

duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$CLIP")
frame_count=$(ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$CLIP")

expected=$(echo "$duration * $EXPECTED_FPS" | bc)
echo "Duration: ${duration}s | Expected frames: ${expected%.*} | Actual frames: ${frame_count}"

# allow +-1 frame tolerance for rounding at clip boundaries
diff=$(echo "$frame_count - ${expected%.*}" | bc)
if (( ${diff#-} <= 1 )); then
  echo "PASS"
else
  echo "FAIL"
fi