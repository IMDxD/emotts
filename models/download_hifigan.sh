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
HIFI_GENERATOR_GDRIVE_FILEID=1XL1vIlglbPucPC75wjja0Jk2oE-hjiwl
HIFI_CONFIG_GDRIVE_FILEID=1DNLZ9IPlgQCmnzYcWchZRHef-V2UTfWv

echo "Downloading HiFiGAN Generator..."
gdown --id $HIFI_GENERATOR_GDRIVE_FILEID --output models/hifi/generator.hifi

echo "Downloading HiFiGAN Config..."
gdown --id $HIFI_CONFIG_GDRIVE_FILEID --output models/hifi/config.json

echo "[Finished] HiFiGAN Vocoder Setup"