HIFI_GENERATOR_GDRIVE_FILEID=1UkTg7vAN1yrJWathlnXg0bMGPdPXvKdR
HIFI_DISCRIMINATOR_GDRIVE_FILEID=1EaY5gp0GdxhEJQpKw15rbjN451UKpY5M
HIFI_CONFIG_GDRIVE_FILEID=1YFXueypxryomZV0sgm2ceJFDlpHjjQDc

echo "Downloading HiFiGAN Generator..."
gdown --id $HIFI_GENERATOR_GDRIVE_FILEID --output models/hifi/generator.hifi

echo "Downloading HiFiGAN Discriminator..."
gdown --id $HIFI_DISCRIMINATOR_GDRIVE_FILEID --output models/hifi/discriminator.hifi

echo "Downloading HiFiGAN Config..."
gdown --id $HIFI_CONFIG_GDRIVE_FILEID --output models/hifi/config.json

echo "[Finished] HiFiGAN Vocoder Setup"