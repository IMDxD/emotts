import pathlib
import re
import subprocess

import torch
from scipy.io.wavfile import write as wav_write

from src.data_process.constanst import MELS_MEAN, MELS_STD
from src.models.feature_models.config import (
    DecoderParams, DurationParams, EncoderParams, GaussianUpsampleParams,
    ModelParams, PostNetParams, RangeParams,
)
from src.models.feature_models.non_attentive_tacotron import (
    NonAttentiveTacotron,
)
from src.models.hifi_gan import load_model as load_hifi
from src.models.hifi_gan.hifi_config import HIFIParams
from src.models.hifi_gan.inference_tensor import inference as hifi_inference
from src.preprocessing.text.cleaners import english_cleaners

SAMPLING_RATE = 22050
N_SPEAKERS = 109
MEL_CHANNELS = 80
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
G2P_OUTPUT_PATH = "predictions/to_g2p.txt"
AUDIO_OUTPUT_PATH = "predictions/generated.wav"
G2P_MODEL_PATH = "models/g2p/english_g2p.zip"
TACOTRON_MODEL_PATH = "models/tacotron/model.pth"
HIFI_PARAMS = HIFIParams(
    dir_path="hifi", config_name="config.json", model_name="generator_v1"
)

PHONEMES_TO_IDS = {
    "<PAD>": 0,
    "<SIL>": 1,
    "IH1": 2,
    "T": 3,
    "W": 4,
    "AH1": 5,
    "Z": 6,
    "AH0": 7,
    "HH": 8,
    "AY1": 9,
    "AE1": 10,
    "M": 11,
    "JH": 12,
    "IH0": 13,
    "S": 14,
    "R": 15,
    "NG": 16,
    "D": 17,
    "UW1": 18,
    "AA1": 19,
    "B": 20,
    "DH": 21,
    "P": 22,
    "AO1": 23,
    "EH2": 24,
    "L": 25,
    "OW2": 26,
    "EH1": 27,
    "SH": 28,
    "ER0": 29,
    "N": 30,
    "EY1": 31,
    "IY0": 32,
    "Y": 33,
    "UH1": 34,
    "K": 35,
    "CH": 36,
    "OY1": 37,
    "V": 38,
    "IY1": 39,
    "OW1": 40,
    "F": 41,
    "AW1": 42,
    "IH2": 43,
    "OW0": 44,
    "TH": 45,
    "IY2": 46,
    "G": 47,
    "ER1": 48,
    "AW2": 49,
    "AY2": 50,
    "EH0": 51,
    "UW0": 52,
    "EY2": 53,
    "AA2": 54,
    "AA0": 55,
    "UW2": 56,
    "ZH": 57,
    "AY0": 58,
    "AE2": 59,
    "AE0": 60,
    "EY0": 61,
    "AH2": 62,
    "AO0": 63,
    "AW0": 64,
    "AO2": 65,
    "UH2": 66,
    "UH0": 67,
    "ER2": 68,
    "OY2": 69,
    "OY0": 70,
}
N_PHONEMES = len(PHONEMES_TO_IDS)


def text_to_file(user_query: str):
    text_path = pathlib.Path("tmp.txt")
    with open(text_path, "w") as fout:
        normalized_content = english_cleaners(user_query)
        normalized_content = " ".join(re.findall("[a-zA-Z]+", normalized_content))
        fout.write(normalized_content)
    subprocess.call(
        ["mfa", "g2p", G2P_MODEL_PATH, text_path.absolute(), G2P_OUTPUT_PATH]
    )
    text_path.unlink()


def parse_g2p(g2p_path: str = G2P_OUTPUT_PATH) -> list:
    with open(g2p_path, "r") as fin:
        phonemes_ids = []
        for line in fin:
            _, word_to_phones = line.rstrip().split("\t", 1)
            phonemes_ids.extend(
                [PHONEMES_TO_IDS[ph] for ph in word_to_phones.split(" ")]
            )
    return phonemes_ids


def get_tacotron_batch(phonemes_ids: list, speaker_id: int = 0, device=DEVICE):
    text_lengths = torch.LongTensor([len(phonemes_ids)])
    phonemes_ids = torch.LongTensor(phonemes_ids).unsqueeze(0).to(device)
    speaker_ids = torch.LongTensor([speaker_id]).to(device)
    return phonemes_ids, text_lengths, speaker_ids


def inference_text_to_speech(
    input_text: str,
    speaker_id: int,
    audio_output_path: str,
    tacotron_model_path: str,
    hifi_config: HIFIParams,
) -> None:
    text_to_file(input_text)
    phoneme_ids = parse_g2p()
    batch = get_tacotron_batch(phoneme_ids, speaker_id, DEVICE)
    tacotron = torch.load(tacotron_model_path, map_location=DEVICE)
    tacotron.eval()
    mels = tacotron.inference(batch)
    mels = mels.permute(0, 2, 1).squeeze(0)
    mels = mels * MELS_STD.to(DEVICE) + MELS_MEAN.to(DEVICE)
    generator = load_hifi(hifi_config, DEVICE)
    audio = hifi_inference(generator, mels, DEVICE).detach().cpu().numpy()
    wav_write(audio_output_path, SAMPLING_RATE, audio)


if __name__ == "__main__":
    inference_text_to_speech(
        input_text="1 ring to rule tham all",
        speaker_id=0,
        audio_output_path=AUDIO_OUTPUT_PATH,
        tacotron_model_path=TACOTRON_MODEL_PATH,
        hifi_config=HIFI_PARAMS,
    )
