## HIFI version 02500000 (original)
#HIFI_GENERATOR_GDRIVE_FILEID=1RkZ8reW0WjR9lE_ztTnN1qFVx24JZhhy
#HIFI_CONFIG_GDRIVE_FILEID=12-XlQw8k52hRT73_hwfQ3bXuIBwegabw

## HIFI version 02600000
#HIFI_GENERATOR_GDRIVE_FILEID=1kQ6avkJ4FDGCEK9QBVXS_qrSoslYF2eo
#HIFI_CONFIG_GDRIVE_FILEID=1YFXueypxryomZV0sgm2ceJFDlpHjjQDc

# HIFI version 02750000
#HIFI_GENERATOR_GDRIVE_FILEID=1KKg9xM70GSZwOFkmZNNXRBQs97JQgUMe
#HIFI_CONFIG_GDRIVE_FILEID=1YFXueypxryomZV0sgm2ceJFDlpHjjQDc

## HIFI version 02900000
#HIFI_GENERATOR_GDRIVE_FILEID=1cPILrDXNw-NNe37F_23SWVl2AMisYMvb
#HIFI_CONFIG_GDRIVE_FILEID=1YFXueypxryomZV0sgm2ceJFDlpHjjQDc

## HIFI version 03000000
#HIFI_GENERATOR_GDRIVE_FILEID=17BfIIHQbvM0tP-QHGhnRXShBG8ac20wr
#HIFI_CONFIG_GDRIVE_FILEID=1YFXueypxryomZV0sgm2ceJFDlpHjjQDc

## HIFI version 02550000 (tuned to 3 frames per step)
EN_HIFI_GENERATOR_GDRIVE_FILEID=1XL1vIlglbPucPC75wjja0Jk2oE-hjiwl
EN_HIFI_CONFIG_GDRIVE_FILEID=1DNLZ9IPlgQCmnzYcWchZRHef-V2UTfWv

### HIFI version 02540000 (russian)
#RU_HIFI_GENERATOR_GDRIVE_FILEID=1ge-ctTXJn2dpuGLrlbWb0FezKzr5Px0f
#RU_HIFI_CONFIG_GDRIVE_FILEID=14j2_Cb0HURB_a_uldbHK3V60Owq3Qsg1

### HIFI version 02600000 (russian)
#RU_HIFI_GENERATOR_GDRIVE_FILEID=1h5phBHQB6DnVWn_DmOeKshCZvXxlm5zT
#RU_HIFI_CONFIG_GDRIVE_FILEID=14j2_Cb0HURB_a_uldbHK3V60Owq3Qsg1

## HIFI version 02700000 (russian)
RU_HIFI_GENERATOR_GDRIVE_FILEID=1SyQVaqewfzzYgxmqS3lpqM8-qzobzDgh
RU_HIFI_CONFIG_GDRIVE_FILEID=14j2_Cb0HURB_a_uldbHK3V60Owq3Qsg1

echo "Downloading HiFiGAN Generator (English)... "
gdown --id $EN_HIFI_GENERATOR_GDRIVE_FILEID --output models/en/hifi/generator.hifi
echo "Downloading HiFiGAN Config (English)..."
gdown --id $EN_HIFI_CONFIG_GDRIVE_FILEID --output models/en/hifi/config.json
echo "[Finished] HiFiGAN Vocoder Setup (English)"

echo "Downloading HiFiGAN Generator (Russian)... "
gdown --id $RU_HIFI_GENERATOR_GDRIVE_FILEID --output models/ru/hifi/generator.hifi
echo "Downloading HiFiGAN Config (Russian)..."
gdown --id $RU_HIFI_CONFIG_GDRIVE_FILEID --output models/ru/hifi/config.json
echo "[Finished] HiFiGAN Vocoder Setup (Russian)"