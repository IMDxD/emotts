#!/usr/bin/env bash
# Version 11-08
#TACOTRON_ZIP_GDRIVE_FILEID=1gEpxOIT7hlKfUyBsTqogU3LPO7wa5Jo9

# Version 11-14
#TACOTRON_ZIP_GDRIVE_FILEID=13du7oJEGuk8orQ9Qd7qn2tYHHjJZyJlT

## Version 11-15 (3 frames per step)
#EN_TACOTRON_ZIP_GDRIVE_FILEID=1NgtqKEQeITfcgo1HPcNwh66jgJqkdaNf

## Version 12-09 (GST)
#EN_TACOTRON_ZIP_GDRIVE_FILEID=1D8xWg70AsjSqaG3LAhtA-FUX9f_C6BxC

# ESD
EN_TACOTRON_ZIP_GDRIVE_FILEID=1vdjFDnNBE8bZXpv-9_pDYsZb073K9VNB

## 225000 naive phones russian
#RU_TACOTRON_ZIP_GDRIVE_FILEID=1KRhsp6cKXHF02Jv24cCKOivoPtaBdord

# 315000 naive phones russian
RU_TACOTRON_ZIP_GDRIVE_FILEID=1mkFB0-UmR84WWM-tWFO-ZEIKeaarQjHc

## 380000 naive phones russian
#RU_TACOTRON_ZIP_GDRIVE_FILEID=1gx7Xbb4AnCByi1OPrq7fe90URfBWRrdj

echo "Downloading Non-Attentive TacoTron model checkpoint (English)..."
gdown --id $EN_TACOTRON_ZIP_GDRIVE_FILEID --output models/en/tacotron/zipped.zip
echo "Extracting TacoTron model and configs... (English)"
unzip -j -o models/en/tacotron/zipped.zip */feature_model.pth */phonemes.json */speakers.json -d models/en/tacotron
# If ESD, also extract HiFi
unzip -j -o models/en/tacotron/zipped.zip *g_* -d models/en/hifi
mv models/en/hifi/*g_* models/en/hifi/generator.hifi
rm -rf models/en/tacotron/zipped.zip
echo "[Finished] Non-Attentive Tacotron Setup (English)"

echo "Downloading Non-Attentive TacoTron model checkpoint (Russian)..."
gdown --id $RU_TACOTRON_ZIP_GDRIVE_FILEID --output models/ru/tacotron/zipped.zip
echo "Extracting TacoTron model and configs... (Russian)"
unzip -j -o models/ru/tacotron/zipped.zip */feature_model.pth */phonemes.json */speakers.json -d models/ru/tacotron
rm -rf models/ru/tacotron/zipped.zip
echo "[Finished] Non-Attentive Tacotron Setup (Russian)"