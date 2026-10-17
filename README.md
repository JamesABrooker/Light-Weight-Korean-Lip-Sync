# Light-Weight Korean Lip-Sync
*This project investigates whether language-specific fine-tuning of the Wav2Lip model can close the cross-lingual lip-sync gap for Korean, or if heavier architecture is actually necessary.*

**Current Status:** In Progress. This project is being completed as part of an academic AIML course at Victoria University of Wellington. The progress of this project can be viewed in the milestones section of the issues tab.

[**Full Design Document & Literature Review**](docs/DesignDoc&LitReview.pdf)

### Overview
Wav2Lip and similar generative lip-sync models are trained and evaluated almost exclusively on English data/metrics, even when the target language isn't English. This has led to performance degradation on phonetically distant languages. Heavier cross-lingual architectures have emerged to close this gap (MuEx, DiffDub), but it has yet to be tested whether this complexity is actually necessary, or whether light-weight, language-specific fine-tuning gets you most of the way there.

This project tests that directly: fine-tuning the Wav2Lip model on Korean data (KMSAV dataset) and comparing against reported MuEx/DiffDub results.

### Approach
1. **Baseline:** Evaluate the unadapted Wav2Lip model on Korean vs. English (using standard LSE-D/LSE-C)
2. **Fine-tune:** Fine-tune the generator only on a 772-clip KMSAV subset, with the sync-expert discriminator frozen
3. **Compare:** Compare the fine-tuned model against both the baseline and reported MuEx/DiffDub numbers
4. **Stretch Goal:** Develop a phoneme-level diagnostic metric to localise failures beyond aggregate sync scores

Full reasoning behind each design decision is in the [design document](docs/DesignDoc&LitReview.pdf).

### Preprocessing
How the data was prepared for Wav2Lip (dataset selection, 25 fps handling, cropping, audio parameters, train/val/test split), including every deviation from the original Wav2Lip pipeline and links to all preprocessing scripts, is documented in the [**Preprocessing log**](logs/milestones/Preprocessing.md). Day-to-day notes are in [`logs/daily/`](logs/daily/).

### Setup
Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on your PATH.

```bash
pip install librosa soundfile scipy numpy matplotlib pandas pyarrow huggingface_hub pytest
git submodule update --init   # pulls Wav2Lip into external/Wav2Lip
pytest tests/ -v              # checks audio preprocessing matches Wav2Lip exactly
```

### Wav2Lip Model

This project uses Wav2Lip (Prajwal et al., ACM MM 2020) for lip-sync inference, used here under its personal/research/non-commercial license.

Pretrained weights (`wav2lip.pth`, `wav2lip_gan.pth`, `lipsync_expert.pth`) are not included. Download them into `checkpoints/`:
`hf download Nekochu/Wav2Lip wav2lip.pth wav2lip_gan.pth lipsync_expert.pth --local-dir checkpoints`

Prajwal, K.R., Mukhopadhyay, R., Namboodiri, V.P., Jawahar, C.V. "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild." ACM Multimedia 2020.

### Korean Dataset

Moved away from the original OLKAVS dataset plan, due to needing to be a Korean national, [view log](logs/daily/2026-08-21.md)

Now using the KMSAV dataset, which uses a Creative Commons licence.

> Kiyoung Park, Changhan Oh, and Sunghee Dong. "KMSAV: Korean Multi-speaker Spontaneous Audio-Visual Speech Recognition Dataset." *ETRI Journal*, 2024.

```bibtex
@article{kmsav,
    title={KMSAV: Korean Multi-speaker Spontaneous Audio-Visual Speech Recognition Dataset},
    author={Park, Kiyoung and Oh, Changhan and Dong, Sunghee},
    journal={ETRI Journal},
    year={2024}
}
```

The KMSAV dataset used in this project is licensed under CC-BY-NC-SA 4.0 by ETRI and is not redistributed here. See [dataset link](https://github.com/etri/kmsav) for terms.

### English Evaluation Dataset

Moved away from the original LRS2 plan, as it requires an institutional licence agreement with the BBC that could not be obtained within the project timeframe, [view log](logs/daily/2026-10-11.md)

Now using LRS3-TED, which shares LRS2's in-the-wild domain and 25 fps frame rate. Only a fixed 100-clip subset of the trainval split is used, for evaluating English performance before and after Korean fine-tuning (clip IDs in `data/lrs3_eval_ids.txt`).

> Triantafyllos Afouras, Joon Son Chung, and Andrew Zisserman. "LRS3-TED: a large-scale dataset for visual speech recognition." *arXiv preprint arXiv:1809.00496*, 2018.

```bibtex
@article{afouras2018lrs3,
    title={LRS3-TED: a large-scale dataset for visual speech recognition},
    author={Afouras, Triantafyllos and Chung, Joon Son and Zisserman, Andrew},
    journal={arXiv preprint arXiv:1809.00496},
    year={2018}
}
```

The LRS3 dataset is licensed under CC BY 4.0. The underlying video content is from TED and TEDx talks and remains subject to TED's terms of use and the CC BY-NC-ND 4.0 licence, so it is used here for non-commercial research only. No LRS3 video or audio, original or generated, is redistributed in this repository. The official LRS3 distribution has been discontinued, so the data was obtained from the verified Hugging Face mirror [TheNHz/ellipsis-lrs3-raw](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw). See the [original licence](https://mm.kaist.ac.kr/datasets/lip_reading/files/license.txt) for terms.

### Author
James Brooker - jamesabrooker@gmail.com

_Final Year BSc Computer Science student at Victoria University of Wellington (Te Herenga Waka)_

_2026 Q3 & Q4_