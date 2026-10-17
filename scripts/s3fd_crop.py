"""
s3fd_crop.py

Re-crops a clip folder (0.jpg, 1.jpg, ..., audio.wav) to tight S3FD face boxes,
matching Wav2Lip's preprocess.py (no padding). Runs on CPU.

Usage (from wav2lip-korean/, in .venv-w2l):
    .venv-w2l/bin/python scripts/s3fd_crop.py <in_dir> <out_dir>
"""
import glob, os, shutil, sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external", "Wav2Lip"))
import face_detection  # noqa: E402


def main(in_dir, out_dir, batch=16):
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                      flip_input=False, device="cpu")
    frames = sorted(glob.glob(os.path.join(in_dir, "*.jpg")),
                    key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    imgs = [cv2.imread(f) for f in frames]
    dets = []
    for i in range(0, len(imgs), batch):
        dets += fa.get_detections_for_batch(np.asarray(imgs[i:i + batch]))

    missing = [i for i, d in enumerate(dets) if d is None]
    if missing:
        sys.exit(f"No face detected in {len(missing)} frames (e.g. {missing[:5]})")

    os.makedirs(out_dir, exist_ok=True)
    for i, (img, (x1, y1, x2, y2)) in enumerate(zip(imgs, dets)):
        cv2.imwrite(os.path.join(out_dir, f"{i}.jpg"), img[y1:y2, x1:x2])
    shutil.copyfile(os.path.join(in_dir, "audio.wav"), os.path.join(out_dir, "audio.wav"))
    print(f"{len(imgs)} frames cropped -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])