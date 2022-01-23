import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


PATHLIKE = Union[str, Path]
FEATURE_MODEL_FILENAME = "feature_model.pth"
MELS_MEAN_FILENAME = "mels_mean.pth"
MELS_STD_FILENAME = "mels_std.pth"
PHONEMES_FILENAME = "phonemes.json"
SPEAKERS_FILENAME = "speakers.json"
CHECKPOINT_DIR = Path("checkpoints")
HIFI_CHECKPOINT_NAME = "hifi"
FEATURE_CHECKPOINT_NAME = "feature"
DATA_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")

RUSSIAN_SPEAKERS = {0: "Игорина"}
try:
    with open("models/en/tacotron/speakers.json", "r") as json_file:
        ENGLISH_SPEAKERS = json.load(json_file)
except FileNotFoundError:
    ENGLISH_SPEAKERS = {0: "Speakers Loading Error"}


@dataclass
class TacoTronCheckpoint:
    path: Path
    model_file_name: str = FEATURE_MODEL_FILENAME
    phonemes_file_name: str = PHONEMES_FILENAME
    speakers_file_name: str = SPEAKERS_FILENAME
    mels_mean_filename: str = MELS_MEAN_FILENAME
    mels_std_filename: str = MELS_STD_FILENAME


@dataclass
class Emotion:
    name: str
    api_name: str
    reference_mels_path: PATHLIKE
    ru_speaker_id: int


@dataclass
class SupportedEmotions:
    angry: Emotion = Emotion(
        name="angry",
        api_name="angry",
        reference_mels_path="Angry.pkl",
        ru_speaker_id=10,
    )
    happy: Emotion = Emotion(
        name="happy",
        api_name="happy",
        reference_mels_path="Happy.pkl",
        ru_speaker_id=21,
    )
    neutral: Emotion = Emotion(
        name="neutral",
        api_name="neutral",
        reference_mels_path="Neutral.pkl",
        ru_speaker_id=13,
    )
    sad: Emotion = Emotion(
        name="sad", api_name="sad", reference_mels_path="Sad.pkl", ru_speaker_id=40
    )
    surprise: Emotion = Emotion(
        name="surprise",
        api_name="surprise",
        reference_mels_path="Surprise.pkl",
        ru_speaker_id=0,
    )
    very_angry: Emotion = Emotion(
        name="very_angry",
        api_name="veryangry",
        reference_mels_path="Very_angry.pkl",
        ru_speaker_id=41,
    )
    very_happy: Emotion = Emotion(
        name="very_happy",
        api_name="veryhappy",
        reference_mels_path="Very_happy.pkl",
        ru_speaker_id=12,
    )


@dataclass
class Language:
    name: str
    api_name: str
    emo_reference_dir: Path
    emo_selector: dict
    speaker_selector: dict
    g2p_model_path: Path
    tacotron_checkpoint: TacoTronCheckpoint
    hifi_params: HIFIParams
    test_phrase: str


@dataclass
class SupportedLanguages:
    english: Language = Language(
        name="English (en-EN)",
        api_name="en",
        emo_reference_dir=Path("models/en/emo_reference"),
        emo_selector={
            "🙂 happy": SupportedEmotions.happy,
            "😲 surprise": SupportedEmotions.surprise,
            "😐 neutral": SupportedEmotions.neutral,
            "😞 sad": SupportedEmotions.sad,
            "😡 angry": SupportedEmotions.angry,
        },
        speaker_selector=ENGLISH_SPEAKERS,
        g2p_model_path=Path("models/en/g2p/english_g2p.zip"),
        tacotron_checkpoint=TacoTronCheckpoint(path=Path("models/en/tacotron")),
        hifi_params=HIFIParams(
            dir_path="en/hifi", config_name="config.json", model_name="generator.hifi"
        ),
        test_phrase="How to fit linear regression?",
    )
    russian: Language = Language(
        name="Russian (ru-RU)",
        api_name="ru",
        emo_reference_dir=Path("models/ru/emo_reference/mels"),
        emo_selector={
            "😃 happy+": SupportedEmotions.very_happy,
            "🙂 happy": SupportedEmotions.happy,
            "😐 neutral": SupportedEmotions.neutral,
            "😞 sad": SupportedEmotions.sad,
            "😒 angry": SupportedEmotions.angry,
            "😡 angry+": SupportedEmotions.very_angry,
        },
        speaker_selector=RUSSIAN_SPEAKERS,
        g2p_model_path=Path("models/ru/g2p/russian_g2p.zip"),
        tacotron_checkpoint=TacoTronCheckpoint(path=Path("models/ru/tacotron")),
        hifi_params=HIFIParams(
            dir_path="ru/hifi", config_name="config.json", model_name="generator.hifi"
        ),
        test_phrase="Я усиленно обогреваю серверную в эти холодные зимние дни",
    )
