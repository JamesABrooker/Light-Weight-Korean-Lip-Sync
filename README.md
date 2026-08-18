# Light-Weight Korean Lip-Sync
*This project investigates whether language-specific fine-tuning of the Wav2Lip model close the cross-lingual lip-sync gap for Korean, or if heavier architecture is actually necessary*

**Current Status:** In Progress. This project is being completed as part of an academic AIML course at Victoria University of Wellington. The progress of this project can be viewed in the milestones section of the issues tab.

[**Full Design Document & Literature Review**](docs/DesignDoc&LitReview.pdf)

### Overview
Wav2Lip and similar generative lip-sync models are trained and evaluated almost exclusively on English data/metrics, even when the target language isn't English. This has lead to performance degradation on phonetically distant languages. Heavier cross-lingual architectures have emerged to close this gap (MuEx, DiffDub), but it has yet to be tested whether this complexity is actually necessary, or whether light-weight, language specific fine-tuning gets you most of the way there.

This project tests that directly; fine-tuning the Wav2Lip model on Korean data (OKLAVS dataset) and comparing against reported MuEx/DiffDub results.

### Approach
1. **Baseline:** Evaluate the unadapted Wav2Lip model on Korean vs. English (using standard LSE-D/LSE-C)
2. **Fine-tune:** Fine-tune the generator only, with a ~500 clip OKLAVS subset, with discriminator frozen
3. **Compare:** Compare the fine tuned model against both baseline and reported MuEx/DiffDub numbers
4. **Stretch Goal:** Develop a phoneme level diagnostic metric to localise failures beyond aggregate sync scores

Full reasoning behind each design decision viewable on the [design document](docs/DesignDoc&LitReview.pdf)

### Author
James Brooker - jamesabrooker@gmail.com

_Final Year BSc Computer Science student at Victoria University of Wellington (Te Herenga Waka)_

_2026 Q3 & Q4_
