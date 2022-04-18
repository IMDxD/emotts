echo -e "\n0) Setup env"
conda env create -n emotts -f russian_preprocessing/environment.yaml
conda activate emotts
conda config --set ssl_verify no
export EMO_DATASET_DIR=/media/diskB/ruslan_a/data/datasets/EMO/russian
export NEUTRAL_DATASET_DIR=/media/diskB/ruslan_a/data/datasets/NEUTRAL/russian
export EMO_OUTPUT_DIR=$EMO_DATASET_DIR/processed_v2
export NEUTRAL_OUTPUT_DIR=$NEUTRAL_DATASET_DIR/processed_v2
export MFA_PREMADE=/media/diskB/ruslan_a/data/datasets/emo_rus_Olga_v2_processed/mfa_espeak_grids

# 16164it [03:36, 74.61it/s]
echo -e "\n1) Prep raw files"
python src/preprocessing/prep_files_russian.py --dataset-dir $EMO_DATASET_DIR/original --text-output-dir $EMO_OUTPUT_DIR/text/raw --audio-output-dir $EMO_OUTPUT_DIR/audio/raw --meta-output-dir $EMO_OUTPUT_DIR/meta
python src/preprocessing/prep_files_russian_neutral.py --dataset-dir $NEUTRAL_DATASET_DIR/original --text-output-dir $NEUTRAL_OUTPUT_DIR/text/raw --audio-output-dir $NEUTRAL_OUTPUT_DIR/audio/raw --meta-output-dir $NEUTRAL_OUTPUT_DIR/meta

# ~1.5-2.0 hours (Deprecated)
# echo -e "\n2) Pausation cutting with VAD"
# python src/preprocessing/pausation_cutting.py --input-dir $EMO_OUTPUT_DIR/audio/raw --output-dir $EMO_OUTPUT_DIR/audio/no_pause --target-sr 96000 --audio-ext wav

# 16071/16071 [02:50<00:00, 94.26it/s]
echo -e "\n3) Resampling and Converting audio to 1-channel"
# python src/preprocessing/resampling.py --input-dir $EMO_OUTPUT_DIR/audio/no_pause --output-dir $EMO_OUTPUT_DIR/audio/resampled --resample-rate 22050 --audio-ext wav
python src/preprocessing/resampling.py --input-dir $EMO_OUTPUT_DIR/audio/raw --output-dir $EMO_OUTPUT_DIR/audio/resampled --resample-rate 22050 --audio-ext wav
python src/preprocessing/resampling.py --input-dir $NEUTRAL_OUTPUT_DIR/audio/raw --output-dir $NEUTRAL_OUTPUT_DIR/audio/resampled --resample-rate 22050 --audio-ext wav

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

# 16071/16071 [13:09<00:00, 20.37it/s]
echo -e "\n4) Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $EMO_OUTPUT_DIR/audio/resampled --output-dir $EMO_OUTPUT_DIR/mels --audio-ext wav
python src/preprocessing/wav_to_mel.py --input-dir $NEUTRAL_OUTPUT_DIR/audio/resampled --output-dir $NEUTRAL_OUTPUT_DIR/mels --audio-ext wav

# 16069/16069 [31:38<00:00,  8.46it/s]
# 16962/16962 [12:22<00:00, 22.85it/s] mix
# Model is needed: https://github.com/snakers4/russian_stt_text_normalization/blob/master/jit_s2s.pt
# Put model to src/preprocessing/text/russian/
echo -e "\n5) Text normalization"
python src/preprocessing/text_normalization_russian.py --input-dir $EMO_OUTPUT_DIR/text/raw --output-dir $EMO_OUTPUT_DIR/mfa_inputs
python src/preprocessing/text_normalization_russian.py --input-dir $NEUTRAL_OUTPUT_DIR/text/raw --output-dir $NEUTRAL_OUTPUT_DIR/mfa_inputs

echo -e "\n6) MFA Alignment setup"

# download a pretrained russian acoustic model, and russian g2p model
mkdir -p models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/russian.zip -P models/mfa
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/g2p/russian_g2p.zip -P models/g2p
# for non-naive phonetization copy from
export RUS_ESPEAK_ACOUSTIC_MODEL=/media/diskB/ruslan_a/models/mfa/rus-espeak-mfa/russian_acoustic_model.zip
export RUS_ESPEAK_LEXICON=/media/diskB/ruslan_a/models/mfa/rus-espeak-mfa/rus-mfa-espeak-lexicon-cleaned.txt


# 16069/16069 [00:00<00:00, 21508.15it/s]
echo -e "\n6.1) Creating word list from dataset"
python src/preprocessing/create_corpus.py --input-dir $EMO_OUTPUT_DIR/text/raw --output-path $EMO_OUTPUT_DIR/meta/words.txt

# 25849/25983 [01:19<00:00, 326.82it/s]
echo -e "\n6.2) Creating G2P lexicon from word list"
mfa g2p -t mfa_tmp -j 32 --clean --overwrite models/g2p/russian_g2p.zip $EMO_OUTPUT_DIR/meta/words.txt models/mfa/russian_lexicon.txt
rm -rf mfa_tmp

# 42it [00:10,  4.01it/s]
# 58it [00:12,  4.54it/s] mix
echo -e "\n7) MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $EMO_OUTPUT_DIR/audio/resampled --output-dir $EMO_OUTPUT_DIR/mfa_inputs

# FINALLY, align phonemes and speech
# ~50 min
echo -e "\n8) MFA Alignment"
echo $EMO_OUTPUT_DIR
mfa align -t mfa_tmp --clean -j 32 $EMO_OUTPUT_DIR/mfa_inputs models/mfa/russian_lexicon.txt models/mfa/russian.zip $EMO_OUTPUT_DIR/mfa_outputs
# for mix
mfa align -t mfa_tmp --clean -j 32 $EMO_OUTPUT_DIR/mfa_inputs $RUS_ESPEAK_LEXICON $RUS_ESPEAK_ACOUSTIC_MODEL $EMO_OUTPUT_DIR/mfa_outputs
rm -rf mfa_tmp

echo -e "\n9) MFA Postprocessing"
# Aggregate mels by speakers
# 16071it [00:00, 16767.84it/s]
# 25429it [00:01, 17870.70it/s] mix
python src/preprocessing/mfa_postprocessing.py --input-dir $EMO_OUTPUT_DIR/mels

# Only for MFA PREMADE case
echo -e "\n10) MFA Premade Preprocessing"
python src/preprocessing/mfa_premade_preprocessing.py --input-dir $MFA_PREMADE --output-dir $EMO_OUTPUT_DIR/mfa_outputs