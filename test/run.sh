#!/bin/bash
conda activate emotts
cd cur_dir

# Download dataset
gdown --id 1vgRmRl4BNrcuVZKrpVL5OSmp34QqEby_ --output vctk.zip

export OUTPUT_DIR=data_test

# Unzip dataset and reorganize folders
unzip -q vctk.zip txt/* wav48_silence_trimmed/*
mkdir -p $OUTPUT_DIR/zip
mv vctk.zip $OUTPUT_DIR/zip
mkdir -p $OUTPUT_DIR/raw/text
mv txt/* $OUTPUT_DIR/raw/text
mkdir -p $OUTPUT_DIR/raw/audio
mv wav48_silence_trimmed/* $OUTPUT_DIR/raw/audio
rm -rf txt wav48_silence_trimmed

echo -e "\n1. Selecting only one mic per speaker"
python src/preprocessing/preprocessing.py --input-dir $OUTPUT_DIR/raw/audio --output-dir $OUTPUT_DIR/processed/audio_single_mic --audio-ext flac

echo -e "\n2. Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/processed/audio_single_mic --output-dir $OUTPUT_DIR/processed/no_pause --target-sr 48000

echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/no_pause --output-dir $OUTPUT_DIR/processed/resampled --resample-rate 22050

echo -e "\n4. Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $OUTPUT_DIR/processed/resampled --output-dir $OUTPUT_DIR/processed/mels

echo -e "\n5. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/raw/text --output-dir $OUTPUT_DIR/processed/mfa_inputs

echo -e "\n6. MFA Alignment setup"
mfa thirdparty download

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
wget -nc -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models
wget -nc -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models

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
# python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/mfa_outputs
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/mels