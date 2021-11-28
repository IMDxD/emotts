#!/bin/bash
conda activate emotts
cd repo

FULL_DATASET_ID=12ZNLXWy5Mezz8A3iMcMoxOei1cPZlzAV

# Download dataset
gdown --id $FULL_DATASET_ID -O msp-improv.zip

export AUDIO_EXT=wav
export TEXT_EXT=txt
export OUTPUT_DIR=data

# Unzip dataset and reorganize folders
echo -e "\nUnzipping data..."
unzip -q msp-improv.zip Audio/* Human_transcriptions.zip -d $OUTPUT_DIR
find $OUTPUT_DIR -name "*.zip" | while read filename; do unzip -q -o -d "`dirname "$filename"`" "$filename"; done;
find $OUTPUT_DIR -name "*.zip" -exec rm {} \;
mkdir -p $OUTPUT_DIR/zip
mv msp-improv.zip $OUTPUT_DIR/zip

mkdir -p $OUTPUT_DIR/raw/text
mv $OUTPUT_DIR/All_human_transcriptions/* $OUTPUT_DIR/raw/text
mkdir -p $OUTPUT_DIR/raw/audio
mv $OUTPUT_DIR/Audio/* $OUTPUT_DIR/raw/audio
rm -rf $OUTPUT_DIR/Audio $OUTPUT_DIR/All_human_transcriptions

echo -e "\n1. Selecting only one mic per speaker"
python src/preprocessing/preprocessing_msp_improv.py --input-text-dir $OUTPUT_DIR/raw/text --input-audio-dir $OUTPUT_DIR/raw/audio --output-text-dir $OUTPUT_DIR/processed/text_reorganized --output-audio-dir $OUTPUT_DIR/processed/audio_reorganized --text-ext $TEXT_EXT --audio-ext $AUDIO_EXT

echo -e "\n2. Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/processed/audio_reorganized --output-dir $OUTPUT_DIR/processed/no_pause --audio-ext $AUDIO_EXT --target-sr 44100

echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/no_pause --output-dir $OUTPUT_DIR/processed/resampled --audio-ext $AUDIO_EXT --resample-rate 22050

echo -e "\n4. Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $OUTPUT_DIR/processed/resampled --output-dir $OUTPUT_DIR/processed/mels --audio-ext $AUDIO_EXT

echo -e "\n5. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/processed/text_reorganized --output-dir $OUTPUT_DIR/processed/mfa_inputs

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