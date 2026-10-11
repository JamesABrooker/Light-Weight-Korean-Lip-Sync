import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath("external/Wav2Lip"))
import audio as w2l_audio

for name in ["english_sample", "korean_sample"]:
    wav = w2l_audio.load_wav(f"tests/fixtures/{name}.wav", 16000)
    np.save(f"tests/fixtures/{name}_mel.npy", w2l_audio.melspectrogram(wav))