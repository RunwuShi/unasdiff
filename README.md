# Unsupervised Single-Channel Audio Separation with Diffusion Source Priors
[![arXiv](https://img.shields.io/badge/arXiv-2512.07226-b31b1b.svg)](https://arxiv.org/abs/2512.07226)
The implementation of paper "Unsupervised Single-Channel Audio Separation with Diffusion Source Priors" (AAAI 2026).


## Introduction
This work presents an unsupervised audio separation framework driven by diffusion source priors, capable of handling speech–sound, sound–sound, and speech–speech separation. We train two diffusion models as source priors for speech and general sound events using the VCTK and FSDKaggle2018 datasets, covering 41 sound classes. Focusing on gradient-based inverse-problem solvers, we further enhance separation performance through a hybrid gradient update schedule and a mixture-informed initialization strategy.

## Setup
To use this repository, please prepare the code, environment and pretrained checkpoints of speech and sound prior models.

1. Prepare the environment,  the model supports flash attention (Version: 2.5.8).
   ```
   $ git clone https://github.com/RunwuShi/unasdiff.git
   $ cd ./unasdiff
   $ conda create -n unasdiff python=3.10
   $ conda activate unasdiff
   $ pip install -r requirements.txt
   ```
2. Download pretrained checkpoints, and place them in `./checkpoints/speech` and `./checkpoints/sound`.
   - Speech souce model: [Google Drive](https://drive.google.com/file/d/1zd7dwY52MvwiyvbxM6kupK8hww6NlSj6/view?usp=drive_link)
   - Sound source model: [Google Drive](https://drive.google.com/file/d/1A9eh4lfrP5m3zGos6xPcG_1r07_Kx7-V/view?usp=drive_link)

## Inference
1. Three separation tasks:
   - Run `test_speech_sound.py` for speech-sound separation. 
   - Run `test_soundevent.py` for sound separation.
   - Run `test_speech_speech.py` for speech separation.

2. For more experiments, please modify settings in `./diffusion/gaussian_diffusion.py`.

   For sound modeling, the sound event test dataset used in this repository is part of FSDKaggle2018, full dataset can be downloaded at https://zenodo.org/records/2552860. The preprocessing file is located at `./sound_dataset_process/out_info.py`. 
   
   The VCTK dataset is used for speech prior modeling, which can be downloaded at https://datashare.ed.ac.uk/handle/10283/3443.

## Remarks
The source-model-based separation approach is not well suited for same-class source separation (e.g. speech separation), because it lacks speaker-conditioning. In future work, we will attempt to address such issue.

## References

- **Undiff**: [arXiv:2306.00721](https://arxiv.org/abs/2306.00721)
- **DPS**: [arXiv:2209.14687](https://arxiv.org/abs/2209.14687), [GitHub](https://github.com/DPS2022/diffusion-posterior-sampling)
- **DSG** [arXiv:2402.03201](https://arxiv.org/abs/2402.03201v1), [GitHub](https://github.com/LingxiaoYang2023/DSG2024)
- **MSDM**: [arXiv:2302.02257](https://arxiv.org/abs/2302.02257), [GitHub](https://github.com/gladia-research-group/multi-source-diffusion-models)
