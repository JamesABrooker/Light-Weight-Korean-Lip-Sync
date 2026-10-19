"""
select_checkpoints.py

For each fine-tuning seed, selects the checkpoint with the lowest validation
sync loss (rule fixed before any test evaluation), copies it to export/ for
transfer, and copies each run's val_log.csv into results/training/ (committed,
for convergence curves). Writes results/training/best_checkpoints.csv.

Usage (from repo root):
    python scripts/select_checkpoints.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--min_step", type=int, default=2, help="ignore the step-1 (pretrained) eval")
    args = p.parse_args()

    export = os.path.join(ROOT, "export")
    train_dir = os.path.join(ROOT, "results", "training")
    os.makedirs(export, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    rows = []
    for s in args.seeds:
        run = os.path.join(ROOT, "runs", f"seed{s}")
        log = os.path.join(run, "val_log.csv")
        with open(log) as f:
            evals = [(int(a), float(b), float(c)) for a, b, c in csv.reader(f)]
        shutil.copyfile(log, os.path.join(train_dir, f"seed{s}_val_log.csv"))

        step, l1, sync = min((e for e in evals if e[0] >= args.min_step), key=lambda e: e[2])
        ckpt = os.path.join(run, f"checkpoint_step{step:09d}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        shutil.copyfile(ckpt, os.path.join(export, f"seed{s}_best.pth"))
        rows.append({"seed": s, "best_step": step, "val_l1": l1, "val_sync_loss": sync,
                     "final_val_sync_loss": evals[-1][2]})
        print(f"seed{s}: best step {step}  val sync {sync:.4f}  (final {evals[-1][2]:.4f})")

    with open(os.path.join(train_dir, "best_checkpoints.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} checkpoints -> {export}")


if __name__ == "__main__":
    main()