import json
import pathlib
import random
import re
import subprocess
from typing import Dict, List, Tuple

import torch
from scipy.io.wavfile import write as wav_write

from src.data_process.constanst import MELS_MEAN, MELS_STD
from src.models.hifi_gan import load_model as load_hifi
from src.models.hifi_gan.hifi_config import HIFIParams
from src.models.hifi_gan.inference_tensor import inference as hifi_inference
from src.models.hifi_gan.meldataset import MAX_WAV_VALUE
from src.preprocessing.text.cleaners import english_cleaners

SAMPLING_RATE = 22050
MEL_CHANNELS = 80
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
G2P_OUTPUT_PATH = "predictions/to_g2p.txt"
AUDIO_OUTPUT_PATH = "predictions/generated.wav"
G2P_MODEL_PATH = "models/g2p/english_g2p.zip"
TACOTRON_MODEL_PATH = "models/tacotron/feature_model.pth"
HIFI_PARAMS = HIFIParams(
    dir_path="hifi", config_name="config.json", model_name="generator.hifi"
)

PHONEME_PATH = "models/tacotron/phonemes.json"
with open(PHONEME_PATH, "r") as json_file:
    PHONEMES_TO_IDS = json.load(json_file)
N_PHONEMES = len(PHONEMES_TO_IDS)
PAUSE_TOKEN = PHONEMES_TO_IDS.get("<SIL>")

SPEAKERS_PATH = "models/tacotron/speakers.json"
with open(SPEAKERS_PATH, "r") as json_file:
    SPEAKERS_TO_IDS = json.load(json_file)
N_SPEAKERS = len(SPEAKERS_TO_IDS)


class CleanedTextIsEmptyStringError(Exception):
    """Raised when input text after cleaning is empty string"""
    pass


def parse_g2p(g2p_path: str = G2P_OUTPUT_PATH) -> Dict[str, list]:
    word_to_phones = {}
    with open(g2p_path, "r") as fin:
        for line in fin:
            word, phones = line.rstrip().split("\t", 1)
            word_to_phones[word] = [PHONEMES_TO_IDS[ph] for ph in phones.split(" ")]
    return word_to_phones


def phonemise(user_query: str) -> List[int]:
    normalized_content = english_cleaners(user_query)
    normalized_content = " ".join(re.findall("[a-zA-Z]+", normalized_content))
    if len(normalized_content) < 1:
        raise CleanedTextIsEmptyStringError
    text_path = pathlib.Path(f"tmp{random.randrange(100000)}.txt")
    with open(text_path, "w") as fout:
        fout.write(normalized_content)
    subprocess.call(
        ["mfa", "g2p", G2P_MODEL_PATH, text_path.absolute(), G2P_OUTPUT_PATH]
    )
    text_path.unlink()
    word_to_phones = parse_g2p()
    phoneme_ids = []
    for word in normalized_content.split(" "):
        phoneme_ids.extend(word_to_phones[word])
    phoneme_ids = [PAUSE_TOKEN] + phoneme_ids + [PAUSE_TOKEN]
    return phoneme_ids


def get_tacotron_batch(
        phonemes_ids: List[int], speaker_id: int = 0, device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
    text_lengths_tensor = torch.LongTensor([len(phonemes_ids)])
    phonemes_ids_tensor = torch.LongTensor(phonemes_ids).unsqueeze(0).to(device)
    speaker_ids_tensor = torch.LongTensor([speaker_id]).to(device)
    return phonemes_ids_tensor, text_lengths_tensor, speaker_ids_tensor


def inference_text_to_speech(
    input_text: str,
    speaker_id: int,
    audio_output_path: str,
    tacotron_model_path: str,
    hifi_config: HIFIParams,
    device: torch.device = DEVICE,
) -> None:
    phoneme_ids = phonemise(input_text)
    batch = get_tacotron_batch(phoneme_ids, speaker_id, device)

    tacotron = torch.load(tacotron_model_path, map_location=device)
    tacotron.to(device)
    tacotron.eval()
    with torch.no_grad():
        mels = tacotron.inference(batch)
        mels = mels.permute(0, 2, 1).squeeze(0)
        mels = mels * MELS_STD.to(device) + MELS_MEAN.to(device)

    generator = load_hifi(hifi_config, device)
    generator.eval()
    with torch.no_grad():
        audio = hifi_inference(generator, mels, device)
        audio = audio * MAX_WAV_VALUE
        audio = audio.type(torch.int16).detach().cpu().numpy()

    wav_write(audio_output_path, SAMPLING_RATE, audio)


if __name__ == "__main__":
    inference_text_to_speech(
        input_text="Two months after receiving his doctorate, Pauli completed the article, which came to 237 pages",
        speaker_id=21,
        audio_output_path=AUDIO_OUTPUT_PATH,
        tacotron_model_path=TACOTRON_MODEL_PATH,
        hifi_config=HIFI_PARAMS,
        device=DEVICE,
    )
