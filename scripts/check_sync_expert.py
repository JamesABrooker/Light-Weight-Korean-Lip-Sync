"""
check_sync_expert.py

Sanity-checks Wav2Lip's frozen sync expert (lipsync_expert.pth) on a real,
in-sync clip: P_sync on aligned audio windows should be clearly higher than on
temporally shifted (off-sync) windows. Input preparation mirrors Wav2Lip's
training code (Tv=5 frames, lower half, 96x96; 16-step mel window at 25 fps).

Usage (from wav2lip-korean/, in .venv-w2l):
    .venv-w2l/bin/python scripts/check_sync_expert.py <clip_dir> \
        --checkpoint checkpoints/lipsync_expert.pth
where <clip_dir> contains 0.jpg, 1.jpg, ... and audio.wav.
"""
import argparse, glob, os, sys
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external", "Wav2Lip"))
import audio  # noqa: E402  (Wav2Lip's own audio.py)
from hparams import hparams  # noqa: E402
from models import SyncNet_color  # noqa: E402

T = 5                     # syncnet_T in Wav2Lip's training scripts
MEL_STEP = 16             # syncnet_mel_step_size
FPS = hparams.fps         # 25
IMG = hparams.img_size    # 96


def load_expert(path, device):
    model = SyncNet_color().to(device)
    ckpt = torch.load(path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def face_window(frames, start):
    win = [cv2.resize(cv2.imread(frames[i]), (IMG, IMG)) for i in range(start, start + T)]
    x = np.concatenate(win, axis=2) / 255.0          # (96, 96, 15)
    x = x.transpose(2, 0, 1)[:, IMG // 2:]           # lower half -> (15, 48, 96)
    return torch.FloatTensor(x)


def mel_window(spec, start_frame):
    i = int(80.0 * (start_frame / float(FPS)))
    m = spec[i:i + MEL_STEP, :]
    if m.shape[0] != MEL_STEP:
        return None
    return torch.FloatTensor(m.T).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip_dir")
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "lipsync_expert.pth"))
    p.add_argument("--offset", type=int, default=10, help="off-sync shift in frames")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_expert(args.checkpoint, device)
    print(f"Loaded sync expert from {args.checkpoint} ({device})")

    frames = sorted(glob.glob(os.path.join(args.clip_dir, "*.jpg")),
                    key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    wav = audio.load_wav(os.path.join(args.clip_dir, "audio.wav"), hparams.sample_rate)
    spec = audio.melspectrogram(wav).T               # (time, 80)

    aligned, shifted = [], []
    with torch.no_grad():
        for t in range(0, len(frames) - T + 1):
            v = face_window(frames, t).unsqueeze(0).to(device)
            for s, store in ((t, aligned), (t + args.offset, shifted)):
                m = mel_window(spec, s)
                if m is None:
                    continue
                a, f = model(m.unsqueeze(0).to(device), v)
                store.append(torch.nn.functional.cosine_similarity(a, f).item())

    n = min(len(aligned), len(shifted))
    aligned, shifted = np.array(aligned[:n]), np.array(shifted[:n])
    print(f"windows: {n}")
    print(f"mean P_sync aligned: {aligned.mean():.3f}")
    print(f"mean P_sync shifted (+{args.offset}f): {shifted.mean():.3f}")
    print(f"aligned > shifted in {np.mean(aligned > shifted) * 100:.1f}% of windows")


if __name__ == "__main__":
    main()