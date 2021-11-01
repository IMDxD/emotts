# MADE 2020-2021 Emotional Text-to-Speech Synthesis

## Basic Architecture
- feature extractor:
    - Attention-based Tacotron 2;
    - Non-Attentive Tacotron (multi-speaker);
- vocoder:
    - pretrained HiFi-GAN (better performance, but needs fine-tuning; 
    
      NOTE: pretrained models work with 22.05 kHz (sample rate) audio only);
     
    [HiFiGAN repo](https://github.com/jik876/hifi-gan)
    
      Input features: 80-dim mel spectral features.
    
    - pretrained LPCNet (worse sound quality, but may be easier to train; 
    
      NOTE: sample rate = 16 kHz (hard coded in the mozilla repo)).
      
    [LPCNet repo by Mozilla](https://github.com/mozilla/LPCNet)
    
      Input features: 20-dim BFC features (18 bark cepstrum coefficients + 2 pitch features (pitch coef and pitch correlation)).
      
NOTE: Downsampling a raw audio is ok (sound quality is slightly lower, but still very decent). Upsampling... might lead to synthesis quality degradation, need to check...
    
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
- text normalization and phonemization: espeak-ng phonemizer (choose American / English accent);

NOTE: The phonetisation actually needs to correspond to the phonetisation used by the aligner. Since you will most likely be using a pretrained aligner (although you may retrain MFA, for example, on your own data), one needs to take the phonemizer that was used for training that model. MFA has some built-in G2P models (G2P: grapheme to phoneme conversion = phonemization). So espeak may not be the optimal choise here, because you will have to retrain MFA / Gentle. Better look into what MFA / Gentle has to offer in terms of phonemizers. NOTE: at training stage MFA does not need a phonemizer, only a lexicon is needed. At inference you will need a standalone phonemizer. 

Normalization: 123 -> hundred and twenty-three, 10/10/2021 -> tenth October two thousand twenty one

Phonemization (like in the dictionary): thought -> θɔːt

- metafile for Tacotron training: audio-id|text-transcript|preprocessed-text-transcript.

Non-Attentive Tacotron specific data preprocessing:
- external phone-level aligner (forced aligner):
    - MFA: output in TextGrid format (convert to durations: `npy`-array of durations in seconds / in frames);
    - Gentle (claimes to be able to align non-verbal emotion expression) (NOTE: for English only).

Aligner (at inference stage for training set preprocessing): takes as input (text graphemic, audio) + lexicon.

Lexicon:
  word \t phonemization_1
  word \t phonemization_2
  
Output from aligner: alignment + phonetization 
    --> extract phonetic transcripts from right side of alignments;
    --> alignments --> durations --> duration loss;
    
--- Tacotron inference ---
  * no durations, no alignemnts, MFA cannot work here;
  * durations are predicted by Tacotron duration predictor;
  * phonetisations need to preducted by another model (phonemizer);

## 2. Basic Emotional Synthesis

### Datasets
- Emo-V-DB (problems with non-speech segments);
- [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) / [MSP-Improv](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html): waiting for reply;
- [MSP-Face](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Face.html) : available for download without any special request (quality to be verified);
- Internal Huawei Russian dataset (cons: 1 speaker, access to only 1 of the 4 team members);
- others (look for data with sample rate of 22 kHz or more).

### Realisation:
- lookup embedding.

## 3. Advanced Emotional Synthesis
- GST-based (global style tokens): prosody transfer model;
- modifications on top of GST.

## Useful Links

### Repositories
- [Tacotron 2](https://github.com/NVIDIA/tacotron2)
- [Non-Annentive Tacotron](https://github.com/Garvit-32/Non-Attentive-Tacotron/)

### Articles
- [28 Nov 2017] [Emotional End-to-End Neural Speech synthesizer](https://arxiv.org/pdf/1711.05447.pdf)
- [16 Feb 2018] [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884)
- [06 Aug 2019] [Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS](https://arxiv.org/pdf/1906.00672)
- [23 Oct 2020] [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646)
- [11 May 2021] [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/pdf/2010.04301)
