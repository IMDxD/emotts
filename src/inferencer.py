import json
from pathlib import Path
from typing import Dict, List, Optional

import tgt
import torch

from src.constants import (
   CHECKPOINT_DIR, FEATURE_MODEL_FILENAME, PHONEMES_FILENAME, SPEAKERS_FILENAME,
)
from src.train_config import load_config
from src.data_process.constanst import MELS_MEAN, MELS_STD
from src.models.feature_models import NonAttentiveTacotron


class Inferencer:

    PAUSE_TOKEN = "<SIL>"
    MFA_PAUSE_TOKEN = ""
    PAD_TOKEN = "<PAD>"
    PHONES_TIER = "phones"
    LEXICON_OOV_TOKEN = "spn"
    MEL_EXT = "pth"
    TACOTRON_DIR = "feature_output"

    def __init__(
        self, config_path: str
    ):
        config = load_config(config_path)
        checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name
        text_data_path = Path(config.data.text_dir)
        data_path = text_data_path.parent
        with open(checkpoint_path / PHONEMES_FILENAME) as f:
            self.phonemes_to_idx: Dict[str, int] = json.load(f)
        with open(checkpoint_path / SPEAKERS_FILENAME) as f:
            self.speakers_to_idx: Dict[str, int] = json.load(f)
        self.device = torch.device(config.device)
        self.feature_model: NonAttentiveTacotron = torch.load(
            checkpoint_path / FEATURE_MODEL_FILENAME, map_location=self.device
        )
        self.text_pathes = text_data_path.rglob(f"*{config.data.text_ext}")
        self.feature_model_mels_path = data_path / self.TACOTRON_DIR
        self.feature_model_mels_path.mkdir(parents=True, exist_ok=True)

    def proceed_data(self) -> None:
        for text_file in self.text_pathes:
            filename = text_file.stem
            speaker = text_file.parent.name
            if speaker not in self.speakers_to_idx:
                continue
            speaker_id = self.speakers_to_idx[speaker]
            phoneme_ids = self.parse_file(text_file)
            if phoneme_ids is None:
                continue
            phonemes_len = len(phoneme_ids)
            batch = (
                torch.LongTensor([phoneme_ids]).to(self.device),
                torch.LongTensor([phonemes_len]),
                torch.LongTensor([speaker_id]).to(self.device),
            )
            output = self.feature_model.inference(batch)
            output = output.permute(0, 2, 1).squeeze(0)
            output = output * MELS_STD.to(self.device) + MELS_MEAN.to(self.device)
            save_dir = self.feature_model_mels_path / speaker
            save_dir.mkdir(exist_ok=True)
            torch.save(
                output, save_dir / f"{filename}.{self.MEL_EXT}"
            )

    def parse_file(self, filepath: Path) -> Optional[List[int]]:
        text_grid = tgt.read_textgrid(filepath)

        if self.PHONES_TIER not in text_grid.get_tier_names():
            return None

        phones_tier = text_grid.get_tier_by_name(self.PHONES_TIER)

        phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]

        if self.LEXICON_OOV_TOKEN in phonemes:
            return None

        phoneme_ids = []
        for phoneme in phonemes:
            if phoneme == self.MFA_PAUSE_TOKEN:
                phoneme = self.PAUSE_TOKEN
            phoneme_ids.append(self.phonemes_to_idx[phoneme])

        return phoneme_ids
