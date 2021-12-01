echo -e "\n0) Setup env"
conda env create -n emotts -f russian_preprocessing/environment.yaml
conda activate emotts
conda config --set ssl_verify no
export RUSSIAN_DATASET_PATH=/media/diskB/ruslan_a/data/datasets/EMO/russian
export OUTPUT_DIR=$RUSSIAN_DATASET_PATH/processed
export OUTPUT_DIR=$RUSSIAN_DATASET_PATH/processed_mix
export MFA_PREMADE=/media/diskB/ruslan_a/data/datasets/emo_rus_Olga_v2_processed/mfa_espeak_grids

# 16164it [03:36, 74.61it/s]
# 25544it [02:45, 154.48it/s] mix
echo -e "\n1) Prep raw files"
# python src/preprocessing/prep_files_russian.py --dataset-dir $RUSSIAN_DATASET_PATH/original --text-output-dir $OUTPUT_DIR/text/raw --audio-output-dir $OUTPUT_DIR/audio/raw
python src/preprocessing/prep_files_russian.py --dataset-dir $RUSSIAN_DATASET_PATH/original_and_neutral --text-output-dir $OUTPUT_DIR/text/raw --audio-output-dir $OUTPUT_DIR/audio/raw

# ~1.5-2.0 hours
echo -e "\n2) Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/audio/raw --output-dir $OUTPUT_DIR/audio/no_pause --target-sr 96000 --audio-ext wav

# 16071/16071 [02:50<00:00, 94.26it/s]
# 25429/25429 [04:18<00:00, 98.26it/s] mix
echo -e "\n3) Resampling and Converting audio to 1-channel"
# python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/audio/no_pause --output-dir $OUTPUT_DIR/audio/resampled --resample-rate 22050 --audio-ext wav
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/audio/raw --output-dir $OUTPUT_DIR/audio/resampled --resample-rate 22050 --audio-ext wav

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

# 16071/16071 [13:09<00:00, 20.37it/s]
# 25429/25429 [13:11<00:00, 32.15it/s] mix
echo -e "\n4) Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $OUTPUT_DIR/audio/resampled --output-dir $OUTPUT_DIR/mels --audio-ext wav

# 16069/16069 [31:38<00:00,  8.46it/s]
# Model is needed: https://github.com/snakers4/russian_stt_text_normalization/blob/master/jit_s2s.pt
# Put model to src/preprocessing/text/russian/
echo -e "\n5) Text normalization"
python src/preprocessing/text_normalization_russian.py --input-dir $OUTPUT_DIR/text/raw --output-dir $OUTPUT_DIR/mfa_inputs

echo -e "\n6) MFA Alignment setup"

# download a pretrained russian acoustic model, and russian g2p model
mkdir -p models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/russian.zip -P models/mfa
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/g2p/russian_g2p.zip -P models/g2p

# 16069/16069 [00:00<00:00, 21508.15it/s]
echo -e "\n6.1) Creating word list from dataset"
python src/preprocessing/create_corpus.py --input-dir $OUTPUT_DIR/text/raw --output-path $OUTPUT_DIR/meta/words.txt

# 25849/25983 [01:19<00:00, 326.82it/s]
echo -e "\n6.2) Creating G2P lexicon from word list"
mfa g2p -t mfa_tmp -j 32 --clean --overwrite models/g2p/russian_g2p.zip $OUTPUT_DIR/meta/words.txt models/mfa/russian_lexicon.txt
rm -rf mfa_tmp

# 42it [00:10,  4.01it/s]
echo -e "\n7) MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/audio/resampled --output-dir $OUTPUT_DIR/mfa_inputs

# FINALLY, align phonemes and speech
# ~50 min
echo -e "\n8) MFA Alignment"
echo $OUTPUT_DIR
mfa align -t mfa_tmp --clean -j 32 $OUTPUT_DIR/mfa_inputs models/mfa/russian_lexicon.txt models/mfa/russian.zip $OUTPUT_DIR/mfa_outputs
rm -rf mfa_tmp

echo -e "\n9) MFA Postprocessing"
# Aggregate mels by speakers
# 16071it [00:00, 16767.84it/s]
# 25429it [00:01, 17870.70it/s] mix
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/mels

# Only for MFA PREMADE case
echo -e "\n10) MFA Premade Preprocessing"
python src/preprocessing/mfa_premade_preprocessing.py --input-dir $MFA_PREMADE --output-dir $OUTPUT_DIR/mfa_outputs