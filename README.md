# Light-Weight Korean Lip-Sync
*This project investigates whether language-specific fine-tuning of the Wav2Lip model close the cross-lingual lip-sync gap for Korean, or if heavier architecture is actually necessary*

**Current Status:** In Progress. This project is being completed as part of an academic AIML course at Victoria University of Wellington. The progress of this project can be viewed in the milestones section of the issues tab.

[**Full Design Document & Literature Review**](docs/DesignDoc&LitReview.pdf)

### Overview
Wav2Lip and similar generative lip-sync models are trained and evaluated almost exclusively on English data/metrics, even when the target language isn't English. This has lead to performance degradation on phonetically distant languages. Heavier cross-lingual architectures have emerged to close this gap (MuEx, DiffDub), but it has yet to be tested whether this complexity is actually necessary, or whether light-weight, language specific fine-tuning gets you most of the way there.

This project tests that directly; fine-tuning the Wav2Lip model on Korean data (KMSAV dataset) and comparing against reported MuEx/DiffDub results.

### Approach
1. **Baseline:** Evaluate the unadapted Wav2Lip model on Korean vs. English (using standard LSE-D/LSE-C)
2. **Fine-tune:** Fine-tune the generator only, with a ~500 clip KMSAV subset, with discriminator frozen
3. **Compare:** Compare the fine tuned model against both baseline and reported MuEx/DiffDub numbers
4. **Stretch Goal:** Develop a phoneme level diagnostic metric to localise failures beyond aggregate sync scores

Full reasoning behind each design decision viewable on the [design document](docs/DesignDoc&LitReview.pdf)

### Dataset

Moved away from original OLKAVS dataset plan, due to needing to be a Korean national, [view log](logs/daily/2026-08-21.md)

Now using KMSAV dataset, which is uses a Creative Commons licence.

> Kiyoung Park, Changhan Oh, and Sunghee Dong. "KMSAV: Korean Multi-speaker Spontaneous Audio-Visual Speech Recognition Dataset." *ETRI Journal*, 2024.

```bibtex
@misc{kmsav,
    title={KMSAV: Korean Multi-speaker Spontaneous Audio-Visual Speech Recognition Dataset},
    author={Kiyoung Park, Changhan Oh and Sunghee Dong},
    year={2024},
    journal={ETRI Journal},
}
```

The KMSAV dataset used in this project is licensed under CC-BY-NC-SA 4.0 by ETRI and is not redistributed here. See [dataset link](https://github.com/etri/kmsav)
for terms.

### Author
James Brooker - jamesabrooker@gmail.com

_Final Year BSc Computer Science student at Victoria University of Wellington (Te Herenga Waka)_

_2026 Q3 & Q4_
