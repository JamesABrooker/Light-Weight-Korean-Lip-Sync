import os
import numpy as np
import pytest
import librosa
from audio_preprocessing import melspectrogram, verify_hparams, HPARAMS
FIX = os.path.join(os.path.dirname(__file__), "fixtures")

@pytest.mark.parametrize("name", ["lrs2_sample", "korean_sample"])
def test_matches_wav2lip(name):
    wav, _ = librosa.load(os.path.join(FIX, f"{name}.wav"), sr=16000)
    ours = melspectrogram(wav)
    ref = np.load(os.path.join(FIX, f"{name}_mel.npy"))
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, atol=1e-4)

def test_value_range():
    wav, _ = librosa.load(os.path.join(FIX, "lrs2_sample.wav"), sr=16000)
    mel = melspectrogram(wav)
    m = HPARAMS["max_abs_value"]
    assert mel.min() >= -m and mel.max() <= m
    assert mel.min() < 0  # catches the [0, 1] normalization bug

def test_hparams():
    verify_hparams()