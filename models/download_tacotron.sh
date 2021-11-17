# Version 11-08
#TACOTRON_ZIP_GDRIVE_FILEID=1gEpxOIT7hlKfUyBsTqogU3LPO7wa5Jo9

# Version 11-14
#TACOTRON_ZIP_GDRIVE_FILEID=13du7oJEGuk8orQ9Qd7qn2tYHHjJZyJlT

# Version 11-15 (3 frames per step)
TACOTRON_ZIP_GDRIVE_FILEID=1NgtqKEQeITfcgo1HPcNwh66jgJqkdaNf

echo "Downloading Non-Attentive TacoTron model checkpoint..."
gdown --id $TACOTRON_ZIP_GDRIVE_FILEID --output models/tacotron/zipped.zip

echo "Extracting TacoTron model and configs..."
unzip models/tacotron/zipped.zip tacotron_vctk_default*/feature_model.pth tacotron_vctk_default*/phonemes.json tacotron_vctk_default*/speakers.json -d models/tacotron
mv models/tacotron/tacotron_vctk_default*/* models/tacotron
rm -rf models/tacotron/zipped.zip models/tacotron/tacotron_vctk_default*/

echo "[Finished] Non-Attentive Tacotron Setup"