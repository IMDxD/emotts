echo -e "\n0) Setup env"
conda env create -n emotts -f russian_preprocessing/environment.yaml
conda activate emotts
conda config --set ssl_verify no
export RUSSIAN_DATASET_PATH=/media/diskB/ruslan_a/data/datasets/EMO/russian/
export OUTPUT_DIR=$RUSSIAN_DATASET_PATH/processed

# 16164it [03:36, 74.61it/s] 
echo -e "\n1) Prep raw files"
python src/preprocessing/prep_files_russian.py --dataset-dir $RUSSIAN_DATASET_PATH/original --text-output-dir $OUTPUT_DIR/text/raw --audio-output-dir $OUTPUT_DIR/audio/raw

echo -e "\n2) Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/audio/raw --output-dir $OUTPUT_DIR/audio/no_pause --target-sr 96000

# 16071/16071 [08:29<00:00, 31.51it/s]
echo -e "\n3) Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/audio/raw --output-dir $OUTPUT_DIR/audio/resampled --resample-rate 22050 --audio-ext wav

# 16071/16071 [01:51<00:00, 143.53it/s]
echo -e "\n4) Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $OUTPUT_DIR/audio/resampled --output-dir $OUTPUT_DIR/mels --audio-ext wav

# 16069/16069 [05:21<00:00, 49.96it/s]
# Model is needed: https://github.com/snakers4/russian_stt_text_normalization/blob/master/jit_s2s.pt
# Put model to src/preprocessing/text/russian/
echo -e "\n5) Text normalization"
python src/preprocessing/text_normalization_russian.py --input-dir $OUTPUT_DIR/text/raw --output-dir $OUTPUT_DIR/mfa_inputs

echo -e "\n6. MFA Alignment setup"

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models
wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

echo -e "\n7. MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/processed/resampled --output-dir $OUTPUT_DIR/processed/mfa_inputs

# FINALLY, align phonemes and speech
echo -e "\n8. MFA Alignment"
echo $OUTPUT_DIR

mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/mfa_inputs models/librispeech-lexicon.txt models/english.zip $OUTPUT_DIR/processed/mfa_outputs
rm -rf temp

echo -e "\n9. MFA Postprocessing"
# Aggregate mels by speakers
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/mels