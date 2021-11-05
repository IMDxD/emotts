## Usage

### Docker

```bash
docker build --rm --tag emotts ./test
docker run --rm -it -v $(pwd):/emotts/cur_dir emotts
```

### Data

```bash
conda create -n emotts python=3.8 pip git click tqdm
conda activate emotts

conda install -y -c conda-forge gdown librosa pysoundfile openblas openfst pynini ngram baumwelch
conda install llvmlite
conda install -y -c pytorch-lts pytorch torchaudio cudatoolkit=10.2  # CUDA 10.2
# conda install -y -c pytorch-lts pytorch torchaudio cpuonly  # CPU


# Download dataset
gdown --id 1vpnvduieFdJSICTNqIqakcOdhn-i_2cU --output vctk.zip

unzip -q vctk.zip txt/* wav48_silence_trimmed/*
mkdir -p data/zip
mv vctk.zip data/zip
mkdir -p data/raw/text
mv txt/* data/raw/text
mkdir -p data/raw/audio
mv wav48_silence_trimmed/* data/raw/audio
rm -rf txt wav48_silence_trimmed


# 1. Selecting only one mic per speaker
python src/preprocessing/preprocessing.py --input-dir data/raw/audio --output-dir data/processed/audio_single_mic --audio-ext flac


# 2. Pausation cutting with VAD
python src/preprocessing/pausation_cutting.py --input-dir data/processed/audio_single_mic --output-dir data/processed/no_pause --target-sr 48000


# 3. Resampling
python src/preprocessing/resampling.py --input-dir data/processed/no_pause --output-dir data/processed/resampled --resample-rate 22050


# 4. Audio to Mel
python src/preprocessing/wav_to_mel.py --input-dir data/processed/resampled --output-dir data/processed/mels

# 5. Text normalization
python src/preprocessing/text_normalization.py --input-dir data/raw/text --output-dir data/processed/mfa_inputs


# 6. Alignment with MFA
pip install montreal-forced-aligner  # install requirements
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git  # install latest updates

mfa thirdparty download

# download a pretrained english acoustic model, and english lexicon
mkdir models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models
wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

# 6.1. Preprocessing
python src/preprocessing/mfa_preprocessing.py --input-dir data/processed/resampled --output-dir data/processed/mfa_inputs

# FINALLY, align phonemes and speech
mfa align -t ./temp --clean -j 4 data/processed/mfa_inputs models/librispeech-lexicon.txt models/english.zip data/processed/mfa_outputs
rm -rf temp

# 7. Postprocessing
# python src/preprocessing/mfa_postprocessing.py --input-dir data/processed/mfa_outputs
python src/preprocessing/mfa_postprocessing.py --input-dir data/processed/mels
```

MODEL_INPUT = (
    INPUT_PHONEMES,
    INPUT_LENGTH,
    INPUT_SPEAKERS,
    INPUT_DURATIONS,
    INPUT_MELS,
)

+ INPUT_PHONEMES: [batch_size x MAX_INPUT_LENGTH_IN_BATCH] (dtype=torch.long)
+ INPUT_LENGTH: [batch_size] (dtype=torch.long)  # number of phonemes
+ INPUT_SPEAKERS: [batch_size] (dtype=torch.long)  # speaker id
+ INPUT_DURATIONS: [batch_size x MAX_INPUT_LENGTH_IN_BATCH] (dtype=torch.long)
- INPUT_MELS: [batch_size x batch_max_total_duration x N_MEL_CHANNELS] (dtype=torch.float)

TextGrid -> INPUT_PHONEMES, INPUT_LENGTH, INPUT_SPEAKERS, INPUT_DURATIONS
.pkl -> INPUT_MELS
