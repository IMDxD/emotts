# MADE 2020-2021 Emotional Text-to-Speech Synthesis

## Basic Architecture
- feature extractor:
    - Attention-based Tacotron 2;
    - Non-Attentive Tacotron (multi-speaker);
- vocoder:
    - pretrained HiFi-GAN (better performance, but needs fine-tuning);
    - pretrained LPCNet (worse sound quality, but may be easier to train).

Attention-based Tacotron: teacher forcing (during training; feed golden previous mel).

## 1. Neutral Synthesis (baseline)

Introduce a metric for emotion synthesis evaluation.

### Datasets
- Single-speaker:
    - **LJSpeech (1st priority, easier to download)**;
    - Nancy (needs an academic request, but corpus quality is better than LJSpeech);
- Multi-speaker:
    - **VCTK (1st priority, but has issues with pauses (see below))**;
    - libritts (more problematic without additional filtration / preprocessing).

### Data Proprocessing
- basic preprocessing: pausation cutting (threshold-based method on spectrogram / VAD (pyvad));
- feature extraction: mel-spectrum (80-dim);
- normalization (channel-wise): standard deviation = 1.0, mean = 0.0;
- text phonemization: espeak-ng phonemizer (choose American / English accent);
- metafile for Tacotron training: audio-id|preprocessed-text-transcript.

Non-Attentive Tacotron specific data preprocessing:
- external phone-level aligner (forced aligner):
    - MFA: output in TextGrid format (convert to durations: `npy`-array of durations in seconds / in frames);
    - Gentle (claimes to be able to align non-verbal emotion expression) (NOTE: for English only).

## 2. Basic Emotional Synthesis

### Datasets
- Emo-V-DB (problems with non-speech segments);
- MSP: waiting for reply;
- Internal Huawei Russian dataset (cons: 1 speaker);
- others (look for data with sample rate of 22 kHz or more).

## 3. Advanced Emotional Synthesis
- lookup embedding;
- GST-based (global style tokens): prosody transfer model;
- modifications on top of GST.

## Useful Links

### Repositories
- [Tacotron 2](https://github.com/NVIDIA/tacotron2)
- [Non-Annentive Tacotron](https://github.com/Garvit-32/Non-Attentive-Tacotron/)

### Articles
- [16 Feb 2018] [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
- [06 Aug 2019] [Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS](https://arxiv.org/abs/1906.00672)
- [23 Oct 2020] [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
- [11 May 2021] [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/abs/2010.04301)
