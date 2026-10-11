"""
Downloads the LRS3 test split from the Hugging Face mirror
(TheNHz/ellipsis-lrs3-raw; official distribution discontinued) and exports a
fixed, seeded subset of clips for English-retention evaluation.

LRS3: Afouras, Chung, Zisserman, arXiv:1809.00496. CC BY 4.0; underlying TED
content under TED terms / CC BY-NC-ND 4.0. Clips are for local evaluation only
and must not be committed or redistributed.

Usage (from wav2lip-korean/):
    python scripts/fetch_lrs3_eval.py --n 100
Requires: huggingface_hub, pyarrow, and `hf auth login` (gated dataset).
"""

import argparse, os, tarfile
from huggingface_hub import HfFileSystem

SRC = "datasets/TheNHz/ellipsis-lrs3-raw/ainncy/trainval.tar.gz"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data", "lrs3_eval"))
    p.add_argument("--ids-file", default=os.path.join(ROOT, "data", "lrs3_eval_ids.txt"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fs = HfFileSystem()
    seen, ids = set(), []
    with fs.open(SRC, "rb", block_size=8 * 1024 * 1024) as fh:
        with tarfile.open(fileobj=fh, mode="r|gz") as tar:
            for m in tar:
                if not m.name.endswith(".mp4"):
                    continue
                parts = m.name.split("/")
                yt_id, clip = parts[-2], os.path.splitext(parts[-1])[0]
                if yt_id in seen:
                    continue # one clip per video
                seen.add(yt_id)
                name = f"{yt_id}_{clip}"
                with open(os.path.join(args.out_dir, f"{name}.mp4"), "wb") as out:
                    out.write(tar.extractfile(m).read())
                ids.append(name)
                if len(ids) >= args.n:
                    break

    with open(args.ids_file, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"{len(ids)} clips -> {args.out_dir}")

if __name__ == "__main__":
    main()