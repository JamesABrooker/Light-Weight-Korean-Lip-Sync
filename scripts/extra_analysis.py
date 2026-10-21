"""
extra_analysis.py

Additional analyses for the report, using existing result files only (no new
inference or training):

  1. Korean breakdown by KMSAV category and by speaker (video): real, baseline
     and fine-tuned LSE per group (tests the language-vs-domain confound,
     e.g. whether eating videos drive the low real-video scores).
  2. Per-clip Spearman correlation between real-video and generated scores on
     the same video frames (does the video, or the speech, drive the scores?).
  3. Per-seed paired Wilcoxon tests vs baseline (Holm within each seed's 4
     tests): how many of the seeds individually show a significant change.
  4. Bootstrap 95% CIs (10,000 clip resamples, seed 0) for the mean
     fine-tuned minus baseline difference.
  5. Boxplot of per-seed means with baseline and real-video reference lines,
     and a real-vs-generated scatter for Korean.

Writes results/extra_analysis.md, results/figures/seed_boxplot.png and
results/figures/real_vs_generated_kmsav.png.

Usage (from wav2lip-korean/, main .venv):
    python scripts/extra_analysis.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
FIG = os.path.join(ROOT, "results", "figures")
MANIFEST = os.path.join(ROOT, "data", "kmsav_pilot_manifest_reviewed.csv")
LANGS = (("kmsav", "Korean"), ("lrs3", "English"))
METRICS = (("lse_d", "LSE-D", +1), ("lse_c", "LSE-C", -1))  # sign: direction of "worse"


# ---------------------------------------------------------------- helpers
def load(label, ds):
    p = os.path.join(MET, f"lse_{label}_{ds}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df = df[df["status"] == "ok"]
    return df.set_index("name")[["lse_d", "lse_c"]].astype(float)


def find_seeds():
    seeds = []
    for f in glob.glob(os.path.join(MET, "lse_finetuned_seed*_kmsav.csv")):
        s = int(re.search(r"seed(\d+)_kmsav", f).group(1))
        if os.path.exists(os.path.join(MET, f"lse_finetuned_seed{s}_lrs3.csv")):
            seeds.append(s)
    return sorted(seeds)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(ps) - k) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def md_table(df, floatfmt="{:.3f}"):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row.values:
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def clip_meta():
    m = pd.read_csv(MANIFEST)
    cat_col = "category" if "category" in m.columns else m.columns[2]
    parts = m["clip_path"].str.replace("\\", "/", regex=False).str.split("/")
    m["name"] = parts.map(lambda p: f"{p[-3]}_{os.path.splitext(p[-1])[0]}")
    m["speaker"] = parts.map(lambda p: p[-3])
    out = m.set_index("name")[[cat_col, "speaker"]]
    return out.rename(columns={cat_col: "category"}), cat_col


# ---------------------------------------------------------------- main
def main():
    seeds = find_seeds()
    os.makedirs(FIG, exist_ok=True)
    lines = ["# Extra analysis\n", f"Fine-tuned seeds: {seeds}\n"]

    data = {}
    for ds, _ in LANGS:
        real, base = load("real", ds), load("baseline_wav2lip", ds)
        ft = {s: load(f"finetuned_seed{s}", ds) for s in seeds}
        common = set(base.index)
        for df in ft.values():
            common &= set(df.index)
        common = sorted(common)
        avg = sum(ft[s].loc[common] for s in seeds) / len(seeds)
        data[ds] = dict(real=real, base=base.loc[common], ft=ft, avg=avg, common=common)

    # 1. Korean breakdown by category and speaker --------------------------
    meta, cat_col = clip_meta()
    k = data["kmsav"]
    df = (meta.join(k["real"].add_prefix("real_"), how="inner")
              .join(k["base"].add_prefix("base_"), how="inner")
              .join(k["avg"].add_prefix("ft_"), how="inner"))
    lines.append(f"\n## 1. Korean test clips by group (n = {len(df)}; category column: `{cat_col}`)\n")
    for key in ("category", "speaker"):
        g = df.groupby(key).agg(
            n=("real_lse_c", "size"),
            real_C=("real_lse_c", "mean"), base_C=("base_lse_c", "mean"), ft_C=("ft_lse_c", "mean"),
            real_D=("real_lse_d", "mean"), base_D=("base_lse_d", "mean"), ft_D=("ft_lse_d", "mean"))
        g["n"] = g["n"].astype(int).astype(str)
        g.index.name = key
        lines += [f"\n### By {key}\n", md_table(g), ""]
    lines.append("C = LSE-C (higher better), D = LSE-D (lower better). ft = mean over seeds.\n")

    # 2. Real vs generated correlation --------------------------------------
    lines.append("\n## 2. Per-clip Spearman correlation, real video vs generated (same video frames)\n")
    lines.append("| Language | Metric | Generated | n | rho | p |\n|---|---|---|---|---|---|")
    for ds, lang in LANGS:
        d = data[ds]
        idx = sorted(set(d["real"].index) & set(d["common"]))
        for col, name, _ in METRICS:
            for gen_label, gen in (("baseline", d["base"]), ("fine-tuned (avg)", d["avg"])):
                rho, p = spearmanr(d["real"].loc[idx, col], gen.loc[idx, col])
                lines.append(f"| {lang} | {name} | {gen_label} | {len(idx)} | {rho:+.3f} | {p:.2g} |")
    lines.append("\nGenerated clips use the same video frames as the real clip but different audio, "
                 "so a strong correlation means the video (pose, quality, framing) drives the scores.\n")

    # 3. Per-seed tests -----------------------------------------------------
    lines.append("\n## 3. Per-seed paired Wilcoxon tests vs baseline (Holm within each seed, 4 tests)\n")
    per_seed = []
    for s in seeds:
        recs = []
        for ds, lang in LANGS:
            d = data[ds]
            f = d["ft"][s].loc[d["common"]]
            for col, name, worse_sign in METRICS:
                diff = (f[col] - d["base"][col]).values
                p = wilcoxon(f[col].values, d["base"][col].values).pvalue
                recs.append(dict(seed=s, lang=lang, metric=name, mean_diff=diff.mean(), p=p,
                                 worse=np.sign(diff.mean()) == worse_sign))
        for r, pa in zip(recs, holm([r["p"] for r in recs])):
            r["p_holm"] = pa
        per_seed += recs
    ps = pd.DataFrame(per_seed)
    lines.append("| Language | Metric | Seeds significantly worse | Seeds significantly better | Seeds n.s. |\n|---|---|---|---|---|")
    for (lang, metric), g in ps.groupby(["lang", "metric"], sort=False):
        sig = g["p_holm"] < 0.05
        lines.append(f"| {lang} | {metric} | {int((sig & g.worse).sum())}/{len(g)} | "
                     f"{int((sig & ~g.worse).sum())}/{len(g)} | {int((~sig).sum())}/{len(g)} |")
    lines.append("")

    # 4. Bootstrap CIs -------------------------------------------------------
    lines.append("\n## 4. Bootstrap 95% CI for mean difference (fine-tuned seed-average minus baseline)\n")
    lines.append("| Language | Metric | Mean diff | 95% CI |\n|---|---|---|---|")
    rng = np.random.default_rng(0)
    for ds, lang in LANGS:
        d = data[ds]
        for col, name, _ in METRICS:
            diff = (d["avg"][col] - d["base"][col]).values
            idx = rng.integers(0, len(diff), size=(10000, len(diff)))
            lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
            lines.append(f"| {lang} | {name} | {diff.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines.append("\nClips resampled with replacement, 10,000 resamples, RNG seed 0.\n")

    # 5a. Boxplot of per-seed means ---------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5))
    for j, (ds, lang) in enumerate(LANGS):
        d = data[ds]
        for i, (col, name, _) in enumerate(METRICS):
            ax = axes[i, j]
            seed_means = [d["ft"][s].loc[d["common"], col].mean() for s in seeds]
            ax.boxplot(seed_means, widths=0.5)
            ax.scatter(np.ones(len(seed_means)) + np.random.default_rng(1).uniform(-0.08, 0.08, len(seed_means)),
                       seed_means, s=12, color="grey", zorder=3)
            ax.axhline(d["base"][col].mean(), color="C3", linestyle="--", label="Baseline")
            if d["real"] is not None:
                ax.axhline(d["real"][col].mean(), color="C2", linestyle=":", label="Real video")
            ax.set_xticks([1])
            ax.set_xticklabels([f"Fine-tuned ({len(seeds)} seeds)"])
            ax.set_title(f"{lang}: {name} ({'lower' if col == 'lse_d' else 'higher'} is better)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    box_path = os.path.join(FIG, "seed_boxplot.png")
    plt.savefig(box_path, dpi=200)
    plt.close()

    # 5b. Real vs generated scatter (Korean LSE-C) -------------------------
    k = data["kmsav"]
    idx = sorted(set(k["real"].index) & set(k["common"]))
    plt.figure(figsize=(4, 3.5))
    plt.scatter(k["real"].loc[idx, "lse_c"], k["base"].loc[idx, "lse_c"], s=10, label="Baseline", alpha=0.7)
    plt.scatter(k["real"].loc[idx, "lse_c"], k["avg"].loc[idx, "lse_c"], s=10, label="Fine-tuned (avg)", alpha=0.7)
    lim = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lim, lim, color="grey", linewidth=0.8, linestyle="--")
    plt.xlabel("Real video LSE-C")
    plt.ylabel("Generated LSE-C")
    plt.title("Korean test clips", fontsize=9)
    plt.legend(fontsize=7)
    plt.tight_layout()
    scatter_path = os.path.join(FIG, "real_vs_generated_kmsav.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    out = os.path.join(ROOT, "results", "extra_analysis.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out}\n-> {box_path}\n-> {scatter_path}")


if __name__ == "__main__":
    main()