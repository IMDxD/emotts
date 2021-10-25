## Usage

### Audio

```bash
conda create -n emotts python=3.8 pip git click tqdm
conda activate emotts


# 0. Selecting only one mic per speaker
python src/preprocessing/01_preprocessing.py --input-dir data/raw/audio --output-dir data/processed/00_audio_single_mic --audio-ext flac


# 1. Pausation cutting with VAD
conda install -y -c conda-forge pysoundfile
conda install -y -c pytorch-lts pytorch torchaudio cudatoolkit=10.2  # CUDA 10.2
# conda install -y -c pytorch-lts pytorch torchaudio cpuonly  # CPU
python src/preprocessing/02_pausation_cutting.py --input-dir data/processed/00_audio_single_mic --output-dir data/processed/01_no_pause --target-sr 48000


# 2. Resampling
python src/preprocessing/03_resampling.py --input-dir data/processed/01_no_pause --output-dir data/processed/02_resampled --resample-rate 22050


# 3. Audio to Mel
python src/preprocessing/04_wav_to_mel.py --input-dir data/processed/02_resampled --output-dir data/processed/03_mels


# 4. Resamling & Normalization
conda install -y -c conda-forge openblas openfst pynini ngram baumwelch
# pip install --no-input openblas openfst pynini ngram baumwelch

pip install montreal-forced-aligner  # install requirements
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git  # install latest updates

mfa thirdparty download

# install sox
sudo apt install -q -y sox

# convert to 16k audio clips
cd data
mkdir audio/wav
# echo "normalize audio clips to sample rate of 16k"
find ./audio/raw -name "*.flac" -type f -execdir sox --norm=-3 {} -r 16k -c 1 `pwd`/audio/wav/{} \;
echo "Number of clips" $(ls ./audio/wav/ | wc -l)


# 5. Alignment with MFA
cd .. && mkdir models && cd models

# download a pretrained english acoustic model, and english lexicon
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip
wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt

cd ..
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts
# FINALLY, align phonemes and speech
mfa align -t ./temp --clean -j 4 data/processed/mfa_inputs models/librispeech-lexicon.txt models/english.zip data/processed/mfa_outputs
```
### Text

```bash
python src/preprocessing/05_text_normalization.py --input-dir data/raw/text --output-dir data/processed/mfa_inputs
```

MODEL_INPUT = (
    INPUT_PHONEMES,
    INPUT_LENGTH,
    INPUT_SPEAKERS,
    INPUT_DURATIONS,
    INPUT_MELS,
)

+ INPUT_PHONEMES: [batch_size x batch_phonemes_max_len] (dtype=torch.long)
+ INPUT_LENGTH: [batch_size] (dtype=torch.long)  # number of phonemes
+ INPUT_SPEAKERS: [batch_size] (dtype=torch.long)  # speaker id
+ INPUT_DURATIONS: [batch_size x batch_phonemes_max_len] (dtype=torch.long)
- INPUT_MELS: [batch_size x batch_max_total_duration x N_MEL_CHANNELS] (dtype=torch.float)

TextGrid -> INPUT_PHONEMES, INPUT_LENGTH, INPUT_SPEAKERS, INPUT_DURATIONS
.pkl -> INPUT_MELS