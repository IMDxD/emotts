from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import tgt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.preprocessing.text.cmudict import valid_symbols
from .config import VCTKDatasetParams

NUMBER = Union[int, float]
PAUSE_TOKEN = "<SIL>"
LEXICON_OOV_TOKEN = "spn"
PHONEME_TO_IDX = {
    phoneme: idx for idx, phoneme in enumerate([PAUSE_TOKEN] + valid_symbols)
}


@dataclass
class VCTKSample:

    phonemes: List[int]
    num_phonemes: int
    speaker_id: int
    durations: List[float]
    mels: torch.Tensor


class VctkDataset(Dataset):

    PHONES_TIER = "phones"
    """Create VCTK Dataset

    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
        * Make sure to put the files as the following structure:
            text
            ├── p225
            |   ├──p225_001.TextGrid
            |   ├──p225_002.TextGrid
            |   └──...
            └── pXXX
                ├──pXXX_YYY.TextGrid
                └──...
            mels
            ├── p225
            |   ├──p225_001.pkl
            |   ├──p225_002.pkl
            |   └──...
            └── pXXX
                ├──pXXX_YYY.pkl
                └──...
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        config: VCTKDatasetParams
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self._text_dir = Path(config.text_dir)
        self._mels_dir = Path(config.mels_dir)
        self._text_ext = config.text_ext
        self._mels_ext = config.mels_ext
        self.speaker_to_idx: Dict[str, int] = {}

        # Check that input dirs exist:
        if not self._text_dir.is_dir():
            raise FileNotFoundError(f'Text data not found at {self._text_dir}.')
        if not self._mels_dir.is_dir():
            raise FileNotFoundError(f'Mels data not found at {self._mels_dir}.')

        # Extracting speaker IDs from the folder structure
        texts = set(
            Path(x.parent.name) / x.stem
            for x in self._text_dir.rglob(f'*{self._text_ext}')
        )
        mels = set(
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f'*{self._mels_ext}')
        )

        self._dataset = []
        self._build_dataset(list(texts & mels))

    def seconds_to_frame(self, seconds: float) -> float:
        return seconds * self.sample_rate / self.hop_size

    def _build_dataset(self, samples):
        speaker_counter = 0
        for sample in tqdm(samples):
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            text_grid = tgt.read_textgrid(tg_path)

            if self.PHONES_TIER not in text_grid.get_tier_names():
                continue

            phones_tier = text_grid.get_tier_by_name(self.PHONES_TIER)

            num_phonemes = len(phones_tier.intervals)
            if sample.parent.name not in self.speaker_to_idx:
                self.speaker_to_idx[sample.parent.name] = speaker_counter
                speaker_counter += 1
            speaker_id = self.speaker_to_idx[sample.parent.name]

            if LEXICON_OOV_TOKEN in [
                x.text for x in phones_tier.get_copy_with_gaps_filled()
            ]:
                continue

            phonemes = [
                PHONEME_TO_IDX[x.text] if x.text else PHONEME_TO_IDX[PAUSE_TOKEN]
                for x in phones_tier.get_copy_with_gaps_filled()
            ]
            durations = [
                self.seconds_to_frame(x.duration())
                for x in phones_tier.get_copy_with_gaps_filled()
            ]

            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            mels: torch.Tensor = torch.load(mels_path)
            mels = mels

            pad_size = mels.shape[-1] - sum(durations)
            # assert pad_size >= 0, f'Expected {mels.shape[-1]} mel frames, got {sum(input_sample["durations"])}'
            # TODO: fix problem when pad_size < 0
            if pad_size < 0:
                # print(f"Removing {-pad_size} frames from input sample duration.")
                durations[-1] -= pad_size
                assert durations[-1] >= 0
            if pad_size > 0:
                phonemes.append(PHONEME_TO_IDX[PAUSE_TOKEN])
                durations.append(pad_size)

            self._dataset.append(VCTKSample(
                phonemes=phonemes,
                num_phonemes=num_phonemes,
                speaker_id=speaker_id,
                durations=durations,
                mels=mels
            ))

        # In DataLoader, we want to put in batch samples
        # with close num_phonemes values ([i:i + batch_size]),
        # so we sort dataset here and never shuffle it afterwards:
        # self._dataset.sort(key=lambda x: x['num_phonemes'])
        self._dataset.sort(key=lambda x: len(x['phonemes']))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


class VctkCollate:
    """
    Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step: int = 1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch: List[VCTKSample]):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [{}, {}, ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x.phonemes) for x in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = input_lengths[0]

        input_speaker_ids = torch.LongTensor(
            [batch[i].speaker_id for i in ids_sorted_decreasing]
        )

        text_padded = torch.zeros((len(batch), max_input_len), dtype=torch.long)
        durations_padded = torch.zeros((len(batch), max_input_len), dtype=torch.long)
        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx].phonemes
            text_padded[i, : len(text)] = torch.as_tensor(text)
            durations = batch[idx].durations
            durations_padded[i, : len(durations)] = torch.as_tensor(durations)

        # Right zero-pad mel-spec
        num_mels = batch[0].mels.squeeze(0).size(0)
        max_target_len = max([x.mels.squeeze(0).size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        for i, idx in enumerate(ids_sorted_decreasing):
            mel = batch[idx].mels.squeeze(0)
            mel_padded[i, :, : mel.size(1)] = mel

        return (
            text_padded,
            input_lengths,
            input_speaker_ids,
            durations_padded,
            mel_padded,
        )
