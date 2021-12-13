import json
import pathlib
import random
import re
import subprocess
from typing import Dict, List, Tuple

import torch
from scipy.io.wavfile import write as wav_write

from src.constants import MELS_MEAN_FILENAME, MELS_STD_FILENAME
from src.models.hifi_gan import load_model as load_hifi
from src.models.hifi_gan.hifi_config import HIFIParams
from src.models.hifi_gan.inference_tensor import inference as hifi_inference
from src.preprocessing.text.cleaners import russian_cleaners

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAMPLING_RATE = 22050


EMO_TO_SPEAKER_ID = {
    "very_angry": 41,
    "angry": 10,
    "sad": 40,
    "happy": 21,
    "very_happy": 12
}

G2P_OUTPUT_PATH = "predictions/to_g2p.txt"
G2P_MODEL_PATH = "models/g2p/russian_g2p.zip"
CHECKPOINT_PATH = pathlib.Path("models/tacotron")
TACOTRON_MODEL_PATH = CHECKPOINT_PATH / "feature_model.pth"
PHONEME_PATH = CHECKPOINT_PATH / "phonemes.json"
HIFI_PARAMS = HIFIParams(
    dir_path="hifi", config_name="config.json", model_name="generator"
)

with open(PHONEME_PATH, "r") as json_file:
    PHONEMES_TO_IDS = json.load(json_file)
N_PHONEMES = len(PHONEMES_TO_IDS)
PAUSE_TOKEN = PHONEMES_TO_IDS.get("")

SPEAKERS_PATH = "models/tacotron/speakers.json"
with open(SPEAKERS_PATH, "r") as json_file:
    SPEAKERS_TO_IDS = json.load(json_file)
N_SPEAKERS = len(SPEAKERS_TO_IDS)

MELS_MEAN = torch.load(CHECKPOINT_PATH / MELS_MEAN_FILENAME)
MELS_STD = torch.load(CHECKPOINT_PATH / MELS_STD_FILENAME)


class CleanedTextIsEmptyStringError(Exception):
    """Raised when input text after cleaning is empty string"""
    pass


def parse_g2p(g2p_path: str) -> Dict[str, list]:
    word_to_phones = {}
    with open(g2p_path, "r") as fin:
        for line in fin:
            word, phones = line.rstrip().split("\t", 1)
            word_to_phones[word] = [PHONEMES_TO_IDS[ph] for ph in phones.split(" ")]
    return word_to_phones


def phonemize(user_query: str) -> List[int]:
    normalized_content = russian_cleaners(user_query)
    normalized_content = " ".join(re.findall("[а-яА-Я]+", normalized_content))
    if len(normalized_content) < 1:
        raise CleanedTextIsEmptyStringError
    text_path = pathlib.Path(f"tmp{random.randrange(100000)}.txt")
    with open(text_path, "w") as fout:
        fout.write(normalized_content)
    subprocess.call(
        ["mfa", "g2p", "-t", "tmp_g2p", G2P_MODEL_PATH, text_path.absolute(), G2P_OUTPUT_PATH]
    )
    text_path.unlink()
    word_to_phones = parse_g2p(G2P_OUTPUT_PATH)
    phoneme_ids = []
    for word in normalized_content.split(" "):
        phoneme_ids.extend(word_to_phones[word])
    phoneme_ids = [PAUSE_TOKEN] + phoneme_ids + [PAUSE_TOKEN]
    return phoneme_ids


def get_tacotron_batch(
        phonemes_ids: List[int], reference: torch.Tensor, speaker_id: int = 0, device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
    text_lengths_tensor = torch.LongTensor([len(phonemes_ids)])
    reference = (reference - MELS_MEAN) / MELS_STD
    reference = reference.permute(0, 2, 1).to(device)
    phonemes_ids_tensor = torch.LongTensor(phonemes_ids).unsqueeze(0).to(device)
    speaker_ids_tensor = torch.LongTensor([speaker_id]).to(device)
    return phonemes_ids_tensor, text_lengths_tensor, speaker_ids_tensor, reference


def inference_text_to_speech(
    input_text: str,
    speaker_id: int,
    audio_output_path: str,
    tacotron_model_path: str,
    hifi_config: HIFIParams,
    reference_path: str,
    device: torch.device = DEVICE,
) -> None:
    phoneme_ids = phonemize(input_text)
    reference = torch.load(reference_path)
    batch = get_tacotron_batch(phoneme_ids, reference, speaker_id, device)

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
        audio = audio * 32768.0
        audio = audio.type(torch.int16).detach().cpu().numpy()

    wav_write(audio_output_path, SAMPLING_RATE, audio)


if __name__ == "__main__":
    emotion = "angry"
    reference_path = f"reference/mels/{emotion}.pkl"
    audio_output_path = f"predictions/{emotion}.wav"
    speaker_id = EMO_TO_SPEAKER_ID[emotion]
    inference_text_to_speech(
        input_text="Быть или не быть? Вот в чем вопрос!",
        speaker_id=speaker_id,
        audio_output_path=audio_output_path,
        tacotron_model_path=TACOTRON_MODEL_PATH,
        hifi_config=HIFI_PARAMS,
        reference_path=reference_path,
        device=DEVICE,
    )
