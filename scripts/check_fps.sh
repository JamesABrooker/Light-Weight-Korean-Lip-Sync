#!/usr/bin/env bash
# check_fps.sh - flag clips with non-25fps or variable frame rate
CSV="$1"

tail -n +2 "$CSV" | while IFS=',' read -r clip_id source_video_id domain split duration clip_path rest; do
  path="$clip_path"
  [ -f "$path" ] || continue

  # avg_frame_rate is the container's nominal rate; r_frame_rate can differ if VFR
  avg_fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$path")
  r_fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$path")

  echo "${clip_id},avg_fps=${avg_fps},r_fps=${r_fps}"
done