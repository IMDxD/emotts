import json
from pathlib import Path
from typing import Dict

import numpy as np
import tgt
import torch
from tqdm import tqdm

from src.constants import (
    CHECKPOINT_DIR, FEATURE_CHECKPOINT_NAME, FEATURE_MODEL_FILENAME,
    MELS_MEAN_FILENAME, MELS_STD_FILENAME, PHONEMES_FILENAME, SPEAKERS_FILENAME,
)
from src.data_process import VCTKBatch
from src.models.feature_models.non_attentive_tacotron import (
    NonAttentiveTacotron,
)
from src.train_config import load_config


class Inferencer:

    PAD_TOKEN = "<PAD>"
    PHONES_TIER = "phones"
    LEXICON_OOV_TOKEN = "spn"
    MEL_EXT = "pth"

    def __init__(
        self, config_path: str
    ):
        config = load_config(config_path)
        checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name / FEATURE_CHECKPOINT_NAME
        with open(checkpoint_path / PHONEMES_FILENAME) as f:
            self.phonemes_to_idx: Dict[str, int] = json.load(f)
        with open(checkpoint_path / SPEAKERS_FILENAME) as f:
            self.speakers_to_idx: Dict[str, int] = json.load(f)
        self.sample_rate = config.sample_rate
        self.hop_size = config.hop_size
        self.device = torch.device(config.device)
        self.feature_model: NonAttentiveTacotron = torch.load(
            checkpoint_path / FEATURE_MODEL_FILENAME, map_location=config.device
        )
        if isinstance(self.feature_model.attention.eps, float):
            self.feature_model.attention.eps = torch.Tensor([self.feature_model.attention.eps])
        self._mels_dir = Path(config.data.mels_dir)
        self._text_dir = Path(config.data.text_dir)
        self._text_ext = config.data.text_ext
        self._mels_ext = config.data.mels_ext
        self.feature_model_mels_path = Path(config.data.feature_dir)
        self.feature_model_mels_path.mkdir(parents=True, exist_ok=True)
        self.mels_mean = torch.load(checkpoint_path / MELS_MEAN_FILENAME)
        self.mels_std = torch.load(checkpoint_path / MELS_STD_FILENAME)

    def seconds_to_frame(self, seconds: float) -> float:
        return seconds * self.sample_rate / self.hop_size

    def proceed_data(self) -> None:
        texts_set = {
            Path(x.parent.name) / x.stem
            for x in self._text_dir.rglob(f"*{self._text_ext}")
        }
        mels_set = {
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f"*{self._mels_ext}")
        }
        samples = list(mels_set & texts_set)
        for sample in tqdm(samples):

            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            text_grid = tgt.read_textgrid(tg_path)

            save_dir = self.feature_model_mels_path / sample.parent.name
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / f"{sample.name}.{self.MEL_EXT}"
            if filepath.exists():
                continue

            if self.PHONES_TIER not in text_grid.get_tier_names():
                continue

            phones_tier = text_grid.get_tier_by_name(self.PHONES_TIER)

            phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]

            if self.LEXICON_OOV_TOKEN in phonemes:
                continue

            speaker_id = self.speakers_to_idx[sample.parent.name]

            phoneme_ids = []
            for phoneme in phonemes:
                phoneme_ids.append(self.phonemes_to_idx[phoneme])

            durations = np.array(
                [
                    self.seconds_to_frame(x.duration())
                    for x in phones_tier.get_copy_with_gaps_filled()
                ],
                dtype=np.float32
            )

            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            mels: torch.Tensor = torch.load(mels_path)
            mels = (mels - self.mels_mean) / self.mels_std

            pad_size = mels.shape[-1] - np.int64(durations.sum())
            if pad_size < 0:
                durations[-1] += pad_size
                assert durations[-1] >= 0
            if pad_size > 0:
                phoneme_ids.append(self.phonemes_to_idx[self.PAD_TOKEN])
                np.append(durations, pad_size)

            with torch.no_grad():
                batch = VCTKBatch(
                    phonemes=torch.LongTensor([phoneme_ids]).to(self.device),
                    num_phonemes=torch.LongTensor([len(phoneme_ids)]),
                    speaker_ids=torch.LongTensor([speaker_id]).to(self.device),
                    durations=torch.FloatTensor([durations]).to(self.device),
                    mels=mels.permute(0, 2, 1).float().to(self.device)
                )
                _, output, _, _, _ = self.feature_model(batch)
                output = output.permute(0, 2, 1).squeeze(0)
                output = output * self.mels_std.to(self.device) + self.mels_mean.to(self.device)

            torch.save(output.float(), filepath)
