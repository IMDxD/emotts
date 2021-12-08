# Version 11-08
#TACOTRON_ZIP_GDRIVE_FILEID=1gEpxOIT7hlKfUyBsTqogU3LPO7wa5Jo9

# Version 11-14
#TACOTRON_ZIP_GDRIVE_FILEID=13du7oJEGuk8orQ9Qd7qn2tYHHjJZyJlT

# Version 11-15 (3 frames per step)
EN_TACOTRON_ZIP_GDRIVE_FILEID=1NgtqKEQeITfcgo1HPcNwh66jgJqkdaNf

# 225000 naive phones russian
RU_TACOTRON_ZIP_GDRIVE_FILEID=1KRhsp6cKXHF02Jv24cCKOivoPtaBdord

echo "Downloading Non-Attentive TacoTron model checkpoint (English)..."
gdown --id $EN_TACOTRON_ZIP_GDRIVE_FILEID --output models/en/tacotron/zipped.zip
echo "Extracting TacoTron model and configs... (English)"
unzip models/en/tacotron/zipped.zip tacotron_vctk_default*/feature_model.pth tacotron_vctk_default*/phonemes.json tacotron_vctk_default*/speakers.json -d models/en/tacotron
mv models/en/tacotron/tacotron_vctk_default*/* models/en/tacotron
rm -rf models/en/tacotron/zipped.zip models/en/tacotron/tacotron_vctk_default*/
echo "[Finished] Non-Attentive Tacotron Setup (English)"

echo "Downloading Non-Attentive TacoTron model checkpoint (Russian)..."
gdown --id $RU_TACOTRON_ZIP_GDRIVE_FILEID --output models/ru/tacotron/zipped.zip
echo "Extracting TacoTron model and configs... (Russian)"
unzip models/ru/tacotron/zipped.zip tacotron_russian*/feature_model.pth tacotron_russian*/phonemes.json tacotron_russian*/speakers.json -d models/ru/tacotron
mv models/ru/tacotron/tacotron_russian*/* models/ru/tacotron
rm -rf models/ru/tacotron/zipped.zip models/ru/tacotron/tacotron_russian*/
echo "[Finished] Non-Attentive Tacotron Setup (Russian)"