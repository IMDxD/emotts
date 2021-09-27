# made-emotts-2021

План:
1. Начать с нейтрального синтеза для бейзлайна (на тройку). Вводим метрику.

Базовая архитектура:
- feature extractor: Attention-based Tacotron / Non-Attentive Tacotron (multi-speaker);
- vocoder: pretrained HifiGAN / pretrained LPCNet.

Attention-based Tacotron: teacher forcing (during training; feed golden previous mel).

Датасеты:
- Single-speaker:
    - LJSpeech (1st priority)
    - Nancy (academic, needs request)
- Multi-speaker:
    - VCTK (1st priority)
    - libritts

Data proprocessing:
    - basic preprocessing: pausation cutting (threshold-based method on spectrogram / VAD (pyvad));
    - feature extraction: mel-spectrum (80-dim);
    - normalization (channel-wise): std = 1, mean = 0;
    - text phonemization: espeak-ng phonemizer (choose American/English accent);
    - reformat dataset: audio-id|preprocessed-text-transcript.

Non-Attentive Tacotron specific data preprocessing:
    - external phone-level aligner (forced aligner);
        - MFA: output in TextGrid format (convert to durations: npy array of durations in seconds / in frames);
        - Gentle (claimes to be able to align non-verbal emotion expression) (NOTE: for English only);

2. Модифицируем архитектуру для эмоционального синтеза.
3. Ещё модифицируем архитектуру для ХОРОШЕГО эмоционального синтеза.

Datasets with emotions: look for data with 22000 sample rate or more.
- Emo-V-DB (problems with non-speech segments)
- MSP: waiting for reply
- Internal Huawei Russian dataset (cons: 1 speaker)

Emotional Speech Synthesis:
- lookup embedding;
- GST-based (global style tokens): prosody transfer model;
- modifications on top of GST;
