# Results summary

Fine-tuned seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


## Korean (KMSAV) (n = 116 paired clips)

| Condition | LSE-D ↓ | LSE-C ↑ |
|---|---|---|
| Real video | 9.517 ± 0.972 | 3.319 ± 1.234 |
| Baseline Wav2Lip | 7.781 ± 0.837 | 5.898 ± 1.254 |
| Fine-tuned (10 seeds) | 7.901 ± 0.101 | 5.449 ± 0.067 |

Baseline/real: mean ± std across clips. Fine-tuned: mean ± std of per-seed means across seeds.

Per-seed means: s0: D=8.007 C=5.369, s1: D=7.813 C=5.532, s2: D=7.781 C=5.495, s3: D=7.836 C=5.528, s4: D=7.885 C=5.464, s5: D=7.932 C=5.448, s6: D=7.876 C=5.445, s7: D=7.825 C=5.466, s8: D=7.940 C=5.431, s9: D=8.114 C=5.315


## English (LRS3) (n = 99 paired clips)

| Condition | LSE-D ↓ | LSE-C ↑ |
|---|---|---|
| Real video | 7.492 ± 0.812 | 6.735 ± 1.432 |
| Baseline Wav2Lip | 7.704 ± 0.897 | 6.057 ± 1.491 |
| Fine-tuned (10 seeds) | 7.686 ± 0.040 | 5.689 ± 0.080 |

Baseline/real: mean ± std across clips. Fine-tuned: mean ± std of per-seed means across seeds.

Per-seed means: s0: D=7.656 C=5.762, s1: D=7.688 C=5.673, s2: D=7.713 C=5.618, s3: D=7.685 C=5.717, s4: D=7.720 C=5.647, s5: D=7.653 C=5.743, s6: D=7.755 C=5.592, s7: D=7.703 C=5.622, s8: D=7.675 C=5.666, s9: D=7.612 C=5.854


## Paired Wilcoxon signed-rank tests (fine-tuned seed-average vs baseline)

| Language | Metric | n | Mean diff | Median diff | Rank-biserial r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| Korean (KMSAV) | LSE-D (lower is better) | 116 | +0.119 | +0.099 | +0.364 | 0.00067 | 0.0013 |
| Korean (KMSAV) | LSE-C (higher is better) | 116 | -0.449 | -0.498 | -0.920 | 8.3e-18 | 3.3e-17 |
| English (LRS3) | LSE-D (lower is better) | 99 | -0.018 | -0.020 | -0.034 | 0.77 | 0.77 |
| English (LRS3) | LSE-C (higher is better) | 99 | -0.368 | -0.390 | -0.615 | 1.1e-07 | 3.2e-07 |

Diff = fine-tuned − baseline. Significance threshold α = 0.05 after Holm correction (4 tests).
