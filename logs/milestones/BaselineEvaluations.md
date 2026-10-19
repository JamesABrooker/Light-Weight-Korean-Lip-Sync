# Baseline Evaluation (M2)

This document records how the unmodified, pretrained Wav2Lip model was evaluated on Korean (KMSAV) and English (LRS3), including setup, verification steps, results, and the audio-visual offset analysis. All evaluation code is in [`scripts/`](../../scripts/), and per-clip results are in [`results/metrics/`](../../results/metrics/).

## Setup

- **Model:** pretrained `wav2lip.pth` (no GAN), unmodified. The current Wav2Lip README links to newer "SD" exports, so the original paper checkpoints (`wav2lip.pth`, `wav2lip_gan.pth`, `lipsync_expert.pth`) were downloaded from the Hugging Face mirror [Nekochu/Wav2Lip](https://huggingface.co/Nekochu/Wav2Lip). Weights are not committed.
- **Environment:** all Wav2Lip and SyncNet code runs in a separate Python 3.10 environment (librosa 0.9.2, torch 2.5.1), as Wav2Lip's code is incompatible with current librosa.
- **Hardware:** inference and scoring ran on CPU (MacBook Pro, Apple M3).
- **Two different SyncNets are used:**
  - *Sync expert* (`lipsync_expert.pth`): Wav2Lip's own discriminator, frozen, provides the sync loss during fine-tuning.
  - *Evaluation SyncNet* ([syncnet_python](https://github.com/joonson/syncnet_python), Chung & Zisserman 2016): computes LSE-D/LSE-C, as in the Wav2Lip paper.

## Verification before evaluation

**Sync expert** ([`scripts/check_sync_expert.py`](../../scripts/check_sync_expert.py)): compares P_sync on the correct audio window against a window shifted by 10 frames, with inputs built exactly as in Wav2Lip's training code (5 frames, lower-half face at 96×96, 16-step mel window).

| Clip | Crop | Aligned | Shifted | Aligned > shifted |
|---|---|---|---|---|
| LRS3 (English) | 224 face track | 0.174 | 0.086 | 62.1% |
| LRS3 (English) | S3FD | 0.789 | 0.082 | 100% |
| KMSAV (Korean) | KMSAV face track | 0.502 | 0.230 | 81.0% |
| KMSAV (Korean) | S3FD | 0.509 | 0.203 | 83.6% |

The expert works correctly on properly framed English video. LRS3's face tracks are framed too loosely for Wav2Lip, so LRS3 is S3FD-cropped. KMSAV face-track crops score the same as S3FD crops, so KMSAV is used as cropped ([`scripts/s3fd_crop.py`](../../scripts/s3fd_crop.py)).

**Evaluation SyncNet:** syncnet_python's demo reproduced the reference output (AV offset 3, min dist 5.352 vs 5.353 expected, confidence 10.087 vs 10.021 expected).

## Evaluation protocol

- **Pairs** ([`scripts/make_eval_pairs.py`](../../scripts/make_eval_pairs.py), seed 0): following the Wav2Lip paper, each test video is paired with audio from a randomly chosen clip of a *different speaker* that is no longer than the video, so the model must generate new lip shapes rather than copy the input. Pairs are fixed in [`filelists/eval_pairs_kmsav.csv`](../../filelists/eval_pairs_kmsav.csv) (116 pairs) and [`filelists/eval_pairs_lrs3.csv`](../../filelists/eval_pairs_lrs3.csv) (100 pairs), 0 fallbacks. Korean video is paired with Korean audio and English with English. The same pairs will be used for every fine-tuned model.
- **Inference** ([`scripts/run_baseline_inference.py`](../../scripts/run_baseline_inference.py)): KMSAV uses a fixed full-frame box (`--box 0 224 0 224`), as clips are already verified face crops. LRS3 uses Wav2Lip's default S3FD face detection. KMSAV inference took ~15 minutes for 116 clips on CPU.
- **Scoring** ([`scripts/score_lse.py`](../../scripts/score_lse.py)): syncnet_python `run_pipeline.py` then `run_syncnet.py` per video. LSE-D = SyncNet min distance, LSE-C = SyncNet confidence, averaged over face tracks per clip. `--min_track 20` is used because syncnet_python's default (100 frames, 4 s) discards most of these short clips. The script also records SyncNet's estimated AV offset.
- **Real-video reference:** the original, unedited test clips are scored the same way (KMSAV video muxed with its own audio), giving a reference row as in the paper's "Real Videos".
- **Runs:** the baseline is a deterministic pretrained model, so it was evaluated once.

## Results

| Condition | LSE-D ↓ | LSE-C ↑ | n |
|---|---|---|---|
| LRS3 real | 7.492 ± 0.812 | 6.735 ± 1.432 | 100 |
| LRS3 Wav2Lip | 7.704 ± 0.897 | 6.057 ± 1.491 | 99 |
| KMSAV real | 9.517 ± 0.972 | 3.319 ± 1.234 | 116 |
| KMSAV Wav2Lip | 7.781 ± 0.837 | 5.898 ± 1.254 | 116 |

Mean ± standard deviation across clips; single deterministic run. One LRS3 clip failed inference and is excluded *(reason: ValueError: Face not detected! (see `logs/inference_baseline_wav2lip_lrs3_*.csv`))*.

**Comparison to the paper (LRS3):** the paper reports Wav2Lip 6.652 / 7.887 and real videos 6.956 / 7.592. Both of our LRS3 rows are worse than the paper's by a similar margin, which suggests our setup (short trainval clips, our crops, `--min_track 20`) is consistently harsher rather than incorrect. Absolute values are therefore not directly comparable to published numbers. Before/after comparisons on our fixed pairs remain valid.

**Observations:**
- Baseline Wav2Lip scores almost identically on Korean and English (LSE-D 7.78 vs 7.70), so the LSE metrics show no cross-lingual gap at baseline.
- Real Korean video scores far worse than both real English video and Wav2Lip's own Korean output (LSE-C 3.32 vs 6.74 and 5.90). SyncNet finds only a weak sync peak on real KMSAV at any offset.

## Audio-visual offset analysis

Because real KMSAV scored so poorly, I checked whether the clips were actually out of sync.

**SyncNet AV offset distribution (real clips):**

| Dataset | Mean ± std (frames) | Counts |
|---|---|---|
| LRS3 | −0.70 ± 0.48 | −1: 68, 0: 31, −2: 1 |
| KMSAV | −1.16 ± 1.73 | −2: 57, −1: 40, +1: 16, 0: 1, −8: 1, +12: 1 |

LRS3, which is known to be well synced, gives this pipeline's in-sync reference point (−1/0). KMSAV is shifted about one frame further on average, with a much wider spread.

**Sync expert offset scan:** P_sync at shifts of −3…+3 frames.

| Shift | LRS3 SY78HMlWo6k_50004 | KMSAV 54ULzusB3zU/000010 |
|---|---|---|
| −3 | 0.094 | 0.332 |
| −2 | 0.167 | 0.423 |
| −1 | 0.444 | **0.546** |
| 0 | **0.775** | 0.499 |
| +1 | 0.470 | 0.356 |
| +2 | 0.166 | 0.292 |
| +3 | 0.107 | 0.268 |

The English clip peaks sharply at 0. The Korean clip peaks at −1. To check whether this was systematic, one training clip from each of 10 different videos was scanned at −1, 0 and +1:

| Best offset | Clips |
|---|---|
| 0 | -_K3g6HyQOg/000233, 0fppXi4h5v8/000013 |
| −1 | 75AWLc2tQIQ/000005, HHNmbwo8kPg/000054 |
| +1 | 1DKwTQ5yL6k/000013, AAAW5DgParc/000008, K10oVqHs5XU/000009, RBo3QndKgrE/000022 |
| No clear peak (P_sync ≈ 0.22–0.25 at all shifts) | 0e76Mv3YWso/000006, Ml2W6PvzCCM/000012 |

Per-video audio-visual offsets of up to ±1 frame (40 ms) were detected in KMSAV, independently by SyncNet (offset distribution) and by the sync expert (offset scan on 10 training clips from different videos: 2 aligned, 2 at −1, 4 at +1, 2 with no measurable sync response). As the direction varied by video, no global correction was applied; per-video offset correction and filtering of non-speech clips are left as future work.

**Interpretation:** LSE-D and LSE-C search across ±15 frames of offset, so a 1-frame shift does not explain real KMSAV's low scores. The low confidence most likely combines (a) SyncNet's weaker response to Korean speech and (b) noisier source data: spontaneous YouTube speech, per-video offsets, and possible non-speech mouth movement (e.g. chewing in the `mukb` category). Language and recording domain are confounded (KMSAV is spontaneous YouTube speech, LRS3 is rehearsed TED talks), so this cannot be attributed to language alone.

## Reproducing

```bash
# from wav2lip-korean/
python scripts/make_eval_pairs.py --seed 0
.venv-w2l/bin/python scripts/run_baseline_inference.py kmsav
.venv-w2l/bin/python scripts/run_baseline_inference.py lrs3
.venv-w2l/bin/python scripts/score_lse.py generated kmsav
.venv-w2l/bin/python scripts/score_lse.py generated lrs3
.venv-w2l/bin/python scripts/score_lse.py real kmsav
.venv-w2l/bin/python scripts/score_lse.py real lrs3
```