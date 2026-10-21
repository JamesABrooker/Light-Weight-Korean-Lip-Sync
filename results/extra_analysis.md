# Extra analysis

Fine-tuned seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


## 1. Korean test clips by group (n = 116; category column: `domain`)


### By category

| category | n | real_C | base_C | ft_C | real_D | base_D | ft_D |
|---|---|---|---|---|---|---|---|
| beau | 13 | 3.569 | 5.244 | 4.743 | 9.660 | 8.234 | 8.488 |
| issu | 27 | 2.274 | 5.266 | 4.780 | 10.492 | 7.960 | 8.111 |
| mukb | 59 | 3.235 | 6.361 | 5.920 | 9.384 | 7.446 | 7.531 |
| scie | 17 | 5.079 | 5.796 | 5.421 | 8.320 | 8.316 | 8.403 |


### By speaker

| speaker | n | real_C | base_C | ft_C | real_D | base_D | ft_D |
|---|---|---|---|---|---|---|---|
| 54ULzusB3zU | 17 | 5.079 | 5.796 | 5.421 | 8.320 | 8.316 | 8.403 |
| 5lX4UInJwBU | 7 | 4.061 | 5.296 | 4.779 | 9.704 | 8.083 | 8.489 |
| ET54EeUHHig | 1 | 3.345 | 7.434 | 7.205 | 8.947 | 6.853 | 6.911 |
| OMJ9i30zBIw | 5 | 3.527 | 6.262 | 5.677 | 9.635 | 7.607 | 7.920 |
| foHGym4mQR8 | 27 | 2.274 | 5.266 | 4.780 | 10.492 | 7.960 | 8.111 |
| h_oWNspfDU0 | 6 | 2.995 | 5.182 | 4.701 | 9.607 | 8.411 | 8.488 |
| qDBBFRFB3-0 | 53 | 3.205 | 6.350 | 5.918 | 9.368 | 7.442 | 7.505 |

C = LSE-C (higher better), D = LSE-D (lower better). ft = mean over seeds.


## 2. Per-clip Spearman correlation, real video vs generated (same video frames)

| Language | Metric | Generated | n | rho | p |
|---|---|---|---|---|---|
| Korean | LSE-D | baseline | 116 | -0.022 | 0.81 |
| Korean | LSE-D | fine-tuned (avg) | 116 | +0.040 | 0.67 |
| Korean | LSE-C | baseline | 116 | +0.108 | 0.25 |
| Korean | LSE-C | fine-tuned (avg) | 116 | +0.121 | 0.2 |
| English | LSE-D | baseline | 99 | -0.101 | 0.32 |
| English | LSE-D | fine-tuned (avg) | 99 | +0.020 | 0.84 |
| English | LSE-C | baseline | 99 | +0.192 | 0.056 |
| English | LSE-C | fine-tuned (avg) | 99 | +0.170 | 0.092 |

Generated clips use the same video frames as the real clip but different audio, so a strong correlation means the video (pose, quality, framing) drives the scores.


## 3. Per-seed paired Wilcoxon tests vs baseline (Holm within each seed, 4 tests)

| Language | Metric | Seeds significantly worse | Seeds significantly better | Seeds n.s. |
|---|---|---|---|---|
| Korean | LSE-D | 6/10 | 0/10 | 4/10 |
| Korean | LSE-C | 10/10 | 0/10 | 0/10 |
| English | LSE-D | 0/10 | 0/10 | 10/10 |
| English | LSE-C | 10/10 | 0/10 | 0/10 |


## 4. Bootstrap 95% CI for mean difference (fine-tuned seed-average minus baseline)

| Language | Metric | Mean diff | 95% CI |
|---|---|---|---|
| Korean | LSE-D | +0.119 | [+0.055, +0.185] |
| Korean | LSE-C | -0.449 | [-0.512, -0.385] |
| English | LSE-D | -0.018 | [-0.126, +0.091] |
| English | LSE-C | -0.368 | [-0.479, -0.256] |

Clips resampled with replacement, 10,000 resamples, RNG seed 0.
