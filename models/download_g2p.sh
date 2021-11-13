G2P_ENGLISH_URL="https://github.com/MontrealCorpusTools/mfa-models/raw/main/g2p/english_g2p.zip"
G2P_ENGLISH_PATH="models/g2p/english_g2p.zip"
G2P_RUSSIAN_URL="https://github.com/MontrealCorpusTools/mfa-models/raw/main/g2p/russian_g2p.zip"
G2P_RUSSIAN_PATH="models/g2p/russian_g2p.zip"

echo Downloading English G2P model...
wget -O $G2P_ENGLISH_PATH $G2P_ENGLISH_URL

echo Downloading Russian G2P model...
wget -O $G2P_RUSSIAN_PATH $G2P_RUSSIAN_URL

echo "[Finished] G2P Phonemizer Setup"