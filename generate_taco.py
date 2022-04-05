import argparse
import json
import random
import re
import subprocess
from pathlib import Path

import torch
from scipy.io.wavfile import write as wav_write

from src.constants import (
    CHECKPOINT_DIR,
    FEATURE_CHECKPOINT_NAME,
    FEATURE_MODEL_FILENAME,
    MELS_MEAN_FILENAME,
    MELS_STD_FILENAME,
    PHONEMES_FILENAME,
    SPEAKERS_FILENAME,
)
from src.models.hifi_gan.models import load_model as load_hifi
from src.preprocessing.text.cleaners import english_cleaners
from src.train_config import load_config

TEXTS = [
    "I can't believe he did it!",
    "He has abandoned all the traditions here.",
    "I mean, can you imagine anything more inappropriate?",
    "That's what the crowd never expected, all of us were astonished to see it.",
    "Now he wonders why people think he is a bit odd.",
    "That is insane!",
]
REMOVE_SPEAKERS = ["p280", "p315", "0019"]
SAVE_PATH = Path("generated_taco")
G2P_MODEL_PATH = "models/g2p/english_g2p.zip"
G2P_OUTPUT_PATH = "predictions/to_g2p.txt"


def text_to_file(user_query: str) -> None:
    text_path = Path("tmp.txt")
    with open(text_path, "w") as fout:
        normalized_content = english_cleaners(user_query)
        normalized_content = " ".join(re.findall("[a-zA-Z]+", normalized_content))
        fout.write(normalized_content)
    subprocess.call(
        ["mfa", "g2p", G2P_MODEL_PATH, text_path.absolute(), G2P_OUTPUT_PATH]
    )
    text_path.unlink()


def parse_g2p(PHONEMES_TO_IDS, g2p_path: str = G2P_OUTPUT_PATH):
    with open(g2p_path, "r") as fin:
        phonemes_ids = []
        phonemes = []
        for line in fin:
            _, word_to_phones = line.rstrip().split("\t", 1)
            phonemes.extend(word_to_phones.split(" "))
            phonemes_ids.extend(
                [PHONEMES_TO_IDS[ph] for ph in word_to_phones.split(" ")]
            )
        phonemes_ids.append(PHONEMES_TO_IDS["<PAD>"])
    return phonemes_ids


def get_tacotron_batch(
    phonemes_ids, reference, speaker_id, device, mels_mean, mels_std
):
    text_lengths_tensor = torch.LongTensor([len(phonemes_ids)])
    reference = (reference - mels_mean) / mels_std
    reference = reference.permute(0, 2, 1).to(device)
    phonemes_ids_tensor = torch.LongTensor(phonemes_ids).unsqueeze(0).to(device)
    speaker_ids_tensor = torch.LongTensor([speaker_id]).to(device)
    return phonemes_ids_tensor, text_lengths_tensor, speaker_ids_tensor, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file path"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config.device)

    vocoder = load_hifi(
        model_path=config.pretrained_hifi,
        hifi_config=config.train_hifi.model_param,
        num_mels=config.n_mels,
        device=device,
    )

    checkpoint_path = Path(
        CHECKPOINT_DIR / config.checkpoint_name / FEATURE_CHECKPOINT_NAME
    )
    mels_mean = torch.load(checkpoint_path / MELS_MEAN_FILENAME)
    mels_std = torch.load(checkpoint_path / MELS_STD_FILENAME)
    with open(checkpoint_path / SPEAKERS_FILENAME) as f:
        speakers_to_id = json.load(f)
    with open(checkpoint_path / PHONEMES_FILENAME) as f:
        phonemes_to_id = json.load(f)

    if config.finetune:
        allowed_speakers = [
            k
            for k in speakers_to_id
            if k not in REMOVE_SPEAKERS and k in config.data.finetune_speakers
        ]
    else:
        allowed_speakers = [
            k
            for k in speakers_to_id
            if k not in REMOVE_SPEAKERS and k not in config.data.finetune_speakers
        ]
    get_speakers = random.choices(allowed_speakers, k=9)

    models = list(checkpoint_path.rglob(f"*_{FEATURE_MODEL_FILENAME}"))
    reference_pathes = Path("references/")
    save_path = SAVE_PATH / config.checkpoint_name

    phonemes_list = []
    for t in TEXTS:
        text_to_file(t)
        phoneme_ids = parse_g2p(phonemes_to_id)
        phonemes_list.append(phoneme_ids)

    for mp in models:
        model = torch.load(mp)
        model.to(device)
        model.eval()
        for speaker in get_speakers:
            for reference in reference_pathes.rglob("*.pkl"):
                emo = reference.stem
                ref_mel = torch.load(reference)
                if config.finetune:
                    speaker_id = speakers_to_id[speaker]
                else:
                    speaker_id = speakers_to_id[reference.parent.name]
                save_folder = save_path / speaker / emo
                save_folder.mkdir(exist_ok=True, parents=True)
                for i, phonemes in enumerate(phonemes_list):
                    batch = get_tacotron_batch(
                        phonemes, ref_mel, speaker_id, device, mels_mean, mels_std
                    )
                    with torch.no_grad():
                        mels = model.inference(batch)
                        mels = mels.permute(0, 2, 1).squeeze(0)
                        mels = mels * mels_std.to(device) + mels_mean.to(device)
                        x = mels.unsqueeze(0)
                        y_g_hat = vocoder(x)
                        audio = y_g_hat.squeeze()
                        audio = audio * 32768
                        audio = audio.type(torch.int16).detach().cpu().numpy()
                        wav_write(save_folder / f"{i + 1}.wav", 22050, audio)


if __name__ == "__main__":
    main()
