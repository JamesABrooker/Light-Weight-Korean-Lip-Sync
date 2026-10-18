"""
make_eval_pairs.py

Builds fixed audio-video pairs for evaluation, following Wav2Lip's benchmark
protocol: each test video is paired with audio from a different, randomly
chosen clip (different speaker) that is no longer than the video, so the model
must generate new lip shapes rather than copy the input.

Writes filelists/eval_pairs_kmsav.csv and filelists/eval_pairs_lrs3.csv.

Usage (from wav2lip-korean/):
    python scripts/make_eval_pairs.py --seed 0
"""
import argparse
import csv
import os
import random

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kmsav_items():
    items = []
    with open(os.path.join(ROOT, "filelists", "kmsav", "test.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vid, utt = line.split("/")
            base = f"kmsav_pilot/cropped/{vid}/utts/{utt}"
            dur = sf.info(os.path.join(ROOT, base + ".wav")).duration
            items.append({"name": f"{vid}_{utt}", "speaker": vid,
                          "video": base + ".mp4", "audio": base + ".wav", "dur": dur})
    return items


def lrs3_items():
    items = []
    with open(os.path.join(ROOT, "data", "lrs3_eval_ids.txt")) as f:
        for line in f:
            cid = line.strip()
            if not cid:
                continue
            wav = f"data/lrs3_eval_processed/wav/{cid}.wav"
            dur = sf.info(os.path.join(ROOT, wav)).duration
            items.append({"name": cid, "speaker": cid.rsplit("_", 1)[0],
                          "video": f"data/lrs3_eval/{cid}.mp4", "audio": wav, "dur": dur})
    return items


def make_pairs(items, rng):
    pairs, n_fallback = [], 0
    for it in items:
        others = [c for c in items if c["speaker"] != it["speaker"]]
        cands = [c for c in others if c["dur"] <= it["dur"]]
        if not cands:  # shortest video: no shorter audio exists
            cands = [min(others, key=lambda c: c["dur"])]
            n_fallback += 1
        a = rng.choice(cands)
        pairs.append({"name": it["name"], "video": it["video"], "audio": a["audio"],
                      "audio_from": a["name"], "video_sec": f"{it['dur']:.2f}",
                      "audio_sec": f"{a['dur']:.2f}"})
    return pairs, n_fallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for lang, items in (("kmsav", kmsav_items()), ("lrs3", lrs3_items())):
        pairs, n_fb = make_pairs(items, random.Random(args.seed))
        out = os.path.join(ROOT, "filelists", f"eval_pairs_{lang}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
        print(f"{lang}: {len(pairs)} pairs ({n_fb} fallback) -> {out}")


if __name__ == "__main__":
    main()