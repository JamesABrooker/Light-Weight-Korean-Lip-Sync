"""
analyse_results.py

Summarises baseline vs fine-tuned Wav2Lip on KMSAV (Korean) and LRS3 (English).
- Real / baseline rows: mean ± std across clips.
- Fine-tuned row: mean ± std of per-seed means across seeds (n = seeds).
- Paired Wilcoxon signed-rank test per language and metric: per-clip
  seed-averaged fine-tuned score vs baseline on the same evaluation pairs,
  with Holm correction across the 4 tests and rank-biserial effect size.
- Validation sync-loss curves for all seeds.

Writes results/summary.md and results/figures/val_sync_loss.png.

Usage (from wav2lip-korean/, main .venv):
    python scripts/analyse_results.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
METRICS = [("lse_d", "LSE-D", "lower"), ("lse_c", "LSE-C", "higher")]


def load(label, ds):
    p = os.path.join(MET, f"lse_{label}_{ds}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df = df[df["status"] == "ok"]
    return df.set_index("name")[["lse_d", "lse_c"]].astype(float)


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(ps) - k) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def find_seeds():
    seeds = []
    for f in glob.glob(os.path.join(MET, "lse_finetuned_seed*_kmsav.csv")):
        s = int(re.search(r"seed(\d+)_kmsav", f).group(1))
        if os.path.exists(os.path.join(MET, f"lse_finetuned_seed{s}_lrs3.csv")):
            seeds.append(s)
    return sorted(seeds)


def main():
    seeds = find_seeds()
    lines = [f"# Results summary\n", f"Fine-tuned seeds: {seeds}\n"]
    tests = []

    for ds, lang in (("kmsav", "Korean (KMSAV)"), ("lrs3", "English (LRS3)")):
        real, base = load("real", ds), load("baseline_wav2lip", ds)
        ft = {s: load(f"finetuned_seed{s}", ds) for s in seeds}
        common = set(base.index)
        for df in ft.values():
            common &= set(df.index)
        common = sorted(common)

        lines += [f"\n## {lang} (n = {len(common)} paired clips)\n",
                  "| Condition | LSE-D ↓ | LSE-C ↑ |", "|---|---|---|"]
        fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
        if real is not None:
            lines.append(f"| Real video | {fmt(real.lse_d.mean(), real.lse_d.std())} | "
                         f"{fmt(real.lse_c.mean(), real.lse_c.std())} |")
        b = base.loc[common]
        lines.append(f"| Baseline Wav2Lip | {fmt(b.lse_d.mean(), b.lse_d.std())} | "
                     f"{fmt(b.lse_c.mean(), b.lse_c.std())} |")
        seed_means = pd.DataFrame({s: ft[s].loc[common].mean() for s in seeds}).T
        lines.append(f"| Fine-tuned ({len(seeds)} seeds) | "
                     f"{fmt(seed_means.lse_d.mean(), seed_means.lse_d.std())} | "
                     f"{fmt(seed_means.lse_c.mean(), seed_means.lse_c.std())} |")
        lines.append("\nBaseline/real: mean ± std across clips. Fine-tuned: mean ± std of per-seed means across seeds.\n")
        lines.append("Per-seed means: " + ", ".join(
            f"s{s}: D={seed_means.loc[s, 'lse_d']:.3f} C={seed_means.loc[s, 'lse_c']:.3f}" for s in seeds) + "\n")

        avg = sum(ft[s].loc[common] for s in seeds) / len(seeds)
        for col, name, better in METRICS:
            d = (avg[col] - b[col]).values
            p = wilcoxon(avg[col].values, b[col].values, alternative="two-sided").pvalue
            tests.append(dict(lang=lang, metric=name, better=better, n=len(common),
                              mean_diff=d.mean(), median_diff=np.median(d),
                              r=rank_biserial(d), p=p))

    adj = holm([t["p"] for t in tests])
    lines += ["\n## Paired Wilcoxon signed-rank tests (fine-tuned seed-average vs baseline)\n",
              "| Language | Metric | n | Mean diff | Median diff | Rank-biserial r | p | p (Holm) |",
              "|---|---|---|---|---|---|---|---|"]
    for t, pa in zip(tests, adj):
        lines.append(f"| {t['lang']} | {t['metric']} ({t['better']} is better) | {t['n']} | "
                     f"{t['mean_diff']:+.3f} | {t['median_diff']:+.3f} | {t['r']:+.3f} | "
                     f"{t['p']:.2g} | {pa:.2g} |")
    lines.append("\nDiff = fine-tuned − baseline. Significance threshold α = 0.05 after Holm correction (4 tests).\n")

    # validation curves
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    curves = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "training", "seed*_val_log.csv"))):
        v = pd.read_csv(f, header=None, names=["step", "l1", "sync"])
        plt.plot(v.step, v.sync, color="grey", alpha=0.4, linewidth=1)
        curves.append(v.set_index("step").sync)
    if curves:
        mean = pd.concat(curves, axis=1).mean(axis=1)
        plt.plot(mean.index, mean.values, color="C0", linewidth=2, label=f"mean ({len(curves)} seeds)")
    plt.xlabel("Training step")
    plt.ylabel("Validation sync loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "val_sync_loss.png"), dpi=200)

    out = os.path.join(ROOT, "results", "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out}\n-> {os.path.join(fig_dir, 'val_sync_loss.png')}")


if __name__ == "__main__":
    main()