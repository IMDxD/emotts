import json
import pathlib
import uuid
import re
import subprocess
from typing import Dict, List, Tuple

import torch
from scipy.io.wavfile import write as wav_write

from src.constants import SupportedLanguages, SupportedEmotions, Emotion, Language
from src.models.hifi_gan import load_model as load_hifi
from src.models.hifi_gan.inference_tensor import inference as hifi_inference
from src.preprocessing.text.cleaners import english_cleaners, russian_cleaners


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAMPLING_RATE = 22050
MAX_WAV_VALUE = 32768.0


class CleanedTextIsEmptyStringError(Exception):
    """Raised when input text after cleaning is empty string"""
    pass


def parse_g2p(g2p_path: pathlib.Path, phonemes_to_ids: Dict[str, int]) -> Dict[str, list]:
    word_to_phones = {}
    with open(g2p_path.absolute(), "r") as fin:
        for line in fin:
            word, phones = line.rstrip().split("\t", 1)
            word_to_phones[word] = [phonemes_to_ids[ph] for ph in phones.split(" ")]
    return word_to_phones


def phonemize(user_query: str, language: Language, phonemes_to_ids: Dict[str, int]) -> List[int]:
    if language == SupportedLanguages.english:
        normalized_content = english_cleaners(user_query)
        normalized_content = " ".join(re.findall("[a-zA-Z]+", normalized_content))
        pause_token = phonemes_to_ids.get("<SIL>")
    elif language == SupportedLanguages.russian:
        normalized_content = russian_cleaners(user_query)
        normalized_content = " ".join(re.findall("[а-яА-Я]+", normalized_content))
        pause_token = phonemes_to_ids.get("")
    else:
        raise NotImplementedError
    if len(normalized_content) < 1:
        raise CleanedTextIsEmptyStringError

    text_path = pathlib.Path(f"cleaned-text-{uuid.uuid4()}.txt")
    g2p_output_path = pathlib.Path(f"g2p-{uuid.uuid4()}.txt")

    with open(text_path, "w") as fout:
        fout.write(normalized_content)
    subprocess.call(
        ["mfa", "g2p", "-t", "g2p_tmp", language.g2p_model_path, text_path, g2p_output_path]
    )
    word_to_phones = parse_g2p(g2p_output_path, phonemes_to_ids)
    text_path.unlink()
    g2p_output_path.unlink()

    phoneme_ids = []
    for word in normalized_content.split(" "):
        phoneme_ids.extend(word_to_phones[word])
    phoneme_ids = [pause_token] + phoneme_ids + [pause_token]

    return phoneme_ids


def get_tacotron_batch(
        phonemes_ids: List[int],
        reference: torch.Tensor,
        speaker_id: int,
        mels_mean: torch.FloatTensor,
        mels_std: torch.FloatTensor,
        device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    text_lengths_tensor = torch.LongTensor([len(phonemes_ids)])
    reference = (reference - mels_mean) / mels_std
    reference = reference.permute(0, 2, 1).to(device)
    phonemes_ids_tensor = torch.LongTensor(phonemes_ids).unsqueeze(0).to(device)
    speaker_ids_tensor = torch.LongTensor([speaker_id]).to(device)
    return phonemes_ids_tensor, text_lengths_tensor, speaker_ids_tensor, reference


def inference_text_to_speech(
    language: Language,
    input_text: str,
    emotion: Emotion,
    audio_output_path: pathlib.Path,
    device: torch.device = DEVICE,
) -> None:

    hifi_config = language.hifi_params
    tacotron_model_path = language.tacotron_checkpoint.path / language.tacotron_checkpoint.model_file_name

    phonemes_path = language.tacotron_checkpoint.path / language.tacotron_checkpoint.phonemes_file_name
    with open(phonemes_path, "r") as json_file:
        phonemes_to_ids = json.load(json_file)

    speakers_path = language.tacotron_checkpoint.path / language.tacotron_checkpoint.speakers_file_name
    with open(speakers_path, "r") as json_file:
        speakers_to_ids = json.load(json_file)

    mels_mean_path = language.tacotron_checkpoint.path / language.tacotron_checkpoint.mels_mean_filename
    mels_mean = torch.load(mels_mean_path)
    mels_std_path = language.tacotron_checkpoint.path / language.tacotron_checkpoint.mels_std_filename
    mels_std = torch.load(mels_std_path)

    if language == SupportedLanguages.english:
        speaker_id = emotion.en_speaker_id
    elif language == SupportedLanguages.russian:
        speaker_id = emotion.ru_speaker_id
    else:
        raise NotImplementedError
    phoneme_ids = phonemize(input_text, language, phonemes_to_ids)
    reference_path = emotion.reference_mels_path
    reference = torch.load(reference_path)
    batch = get_tacotron_batch(phoneme_ids, reference, speaker_id, mels_mean, mels_std, device)

    tacotron = torch.load(tacotron_model_path, map_location=device)
    tacotron.to(device)
    tacotron.eval()
    with torch.no_grad():
        mels = tacotron.inference(batch)
        mels = mels.permute(0, 2, 1).squeeze(0)
        mels = mels * mels_std.to(device) + mels_mean.to(device)

    generator = load_hifi(hifi_config, device)
    generator.eval()
    with torch.no_grad():
        audio = hifi_inference(generator, mels, device)
        audio = audio * MAX_WAV_VALUE
        audio = audio.type(torch.int16).detach().cpu().numpy()

    wav_write(audio_output_path, SAMPLING_RATE, audio)


if __name__ == "__main__":
    inference_text_to_speech(
        language=SupportedLanguages.english,
        input_text="Two months after receiving his doctorate, Pauli completed the article, which came to 237 pages",
        emotion=SupportedEmotions.happy,
        audio_output_path=pathlib.Path("predictions/generated.wav"),
        device=DEVICE,
    )
