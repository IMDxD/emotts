import json
import pathlib
import re
import subprocess
from typing import List, Tuple

import torch
from scipy.io.wavfile import write as wav_write

from src.data_process.constanst import MELS_MEAN, MELS_STD
from src.models.hifi_gan import load_model as load_hifi
from src.models.hifi_gan.hifi_config import HIFIParams
from src.models.hifi_gan.inference_tensor import inference as hifi_inference
from src.preprocessing.text.cleaners import english_cleaners

SAMPLING_RATE = 22050
N_SPEAKERS = 109
MEL_CHANNELS = 80
DEVICE = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
CHECKPOINT_PATH = pathlib.Path("checkpoints/adversarial_gst_3_frame_6_head_03_adv_esd_vctk_no_loss/")
G2P_OUTPUT_PATH = "predictions/to_g2p.txt"
AUDIO_OUTPUT_PATH = "predictions/surprise20.wav"
G2P_MODEL_PATH = "models/g2p/english_g2p.zip"
TACOTRON_MODEL_PATH = CHECKPOINT_PATH / "279_feature_model.pth"
REFERENCE_PATH = "data/esd/mels/0020/0020_001458.pkl"
HIFI_PARAMS = HIFIParams(
    dir_path="cp_hifigan", config_name="config.json", model_name="gen_no_loss"
)

with open(CHECKPOINT_PATH / "phonemes.json") as f:
    PHONEMES_TO_IDS = json.load(f)
    
with open(CHECKPOINT_PATH / "speakers.json") as f:
    SPEAKER_TO_IDS = json.load(f)
    
N_PHONEMES = len(PHONEMES_TO_IDS)


def text_to_file(user_query: str) -> None:
    text_path = pathlib.Path("tmp.txt")
    with open(text_path, "w") as fout:
        normalized_content = english_cleaners(user_query)
        normalized_content = " ".join(re.findall("[a-zA-Z]+", normalized_content))
        fout.write(normalized_content)
    subprocess.call(
        ["mfa", "g2p", G2P_MODEL_PATH, text_path.absolute(), G2P_OUTPUT_PATH]
    )
    text_path.unlink()


def parse_g2p(g2p_path: str = G2P_OUTPUT_PATH) -> List[int]:
    with open(g2p_path, "r") as fin:
        phonemes_ids = [PHONEMES_TO_IDS["<PAD>"]]
        for line in fin:
            _, word_to_phones = line.rstrip().split("\t", 1)
            phonemes_ids.extend(
                [PHONEMES_TO_IDS[ph] for ph in word_to_phones.split(" ")]
            )
        phonemes_ids.append(PHONEMES_TO_IDS["<PAD>"])
    return phonemes_ids


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
    reference_path: str
) -> None:
    text_to_file(input_text)
    phoneme_ids = parse_g2p()
    reference = torch.load(reference_path)
    batch = get_tacotron_batch(phoneme_ids, reference, speaker_id, DEVICE)

    tacotron = torch.load(tacotron_model_path, map_location=DEVICE)
    tacotron.eval()
    with torch.no_grad():
        mels = tacotron.inference(batch)
        mels = mels.permute(0, 2, 1).squeeze(0)
        mels = mels * MELS_STD.to(DEVICE) + MELS_MEAN.to(DEVICE)

    generator = load_hifi(hifi_config, DEVICE)
    generator.eval()
    with torch.no_grad():
        audio = hifi_inference(generator, mels, DEVICE)
        audio = audio * 32768
        audio = audio.type(torch.int16).detach().cpu().numpy()

    wav_write(audio_output_path, SAMPLING_RATE, audio)


if __name__ == "__main__":
    
    with open("references.txt") as f:
        a = f.read()
        
    emodata = [v.split(" - ") for v in a.split("\n")]
    path = pathlib.Path("references")
    for emo, name in emodata:
        speaker = name[:4]
        name = name.strip()
        emo = emo.lower()

        REFERENCE_PATH = f"data/esd/mels/{speaker}/{name}.pkl"
        AUDIO_OUTPUT_PATH = f"predictions/{speaker}_{emo}.wav"

        inference_text_to_speech(
            input_text="How to fit linear regression",
            speaker_id=SPEAKER_TO_IDS[speaker],
            audio_output_path=AUDIO_OUTPUT_PATH,
            tacotron_model_path=TACOTRON_MODEL_PATH,
            hifi_config=HIFI_PARAMS,
            reference_path=REFERENCE_PATH
        )