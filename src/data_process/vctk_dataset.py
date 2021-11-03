import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import tgt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import VCTKDatasetParams
from .constanst import MELS_MEAN, MELS_STD


NUMBER = Union[int, float]


@dataclass
class VCTKSample:

    phonemes: List[int]
    num_phonemes: int
    speaker_id: int
    durations: List[float]
    mels: torch.Tensor


@dataclass
class VCTKBatch:

    phonemes: torch.Tensor
    num_phonemes: torch.Tensor
    speaker_ids: torch.Tensor
    durations: torch.Tensor
    mels: torch.Tensor


class VctkDataset(Dataset):
    def __init__(self, data: List[VCTKSample]):
        self._dataset = data
        self._dataset.sort(key=lambda x: len(x.phonemes))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int) -> VCTKSample:
        return self._dataset[idx]


class VCTKFactory:

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

    PAUSE_TOKEN = "<SIL>"
    MFA_PAUSE_TOKEN = ''
    PAD_TOKEN = "<PAD>"
    LEXICON_OOV_TOKEN = "spn"
    PHONES_TIER = "phones"
    SPEAKER_JSON_NAME = "speakers.json"
    PHONEMES_JSON_NAME = "phonemes.json"

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        config: VCTKDatasetParams,
        phonemes_to_id: Dict[str, int] = None,
        speakers_to_id: Dict[str, int] = None,
    ):

        self._mels_dir = Path(config.mels_dir)
        self._text_dir = Path(config.text_dir)
        self._text_ext = config.text_ext
        self._mels_ext = config.mels_ext
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        if phonemes_to_id:
            self.phoneme_to_id = phonemes_to_id
        else:
            self.phoneme_to_id: Dict[str, int] = {
                self.PAD_TOKEN: 0,
                self.PAUSE_TOKEN: 1,
            }
        if speakers_to_id:
            self.speaker_to_id = speakers_to_id
        else:
            self.speaker_to_id: Dict[str, int] = {}
        self._dataset: List[VCTKSample] = self._build_dataset()

    @staticmethod
    def add_to_mapping(mapping: Dict[str, int], token: str, index: int) -> int:
        if token not in mapping:
            mapping[token] = index
            index += 1
        return index

    def seconds_to_frame(self, seconds: float) -> float:
        return seconds * self.sample_rate / self.hop_size

    def _build_dataset(self) -> List[VCTKSample]:
        speakers_counter = 0
        phonemes_counter = 2
        dataset: List[VCTKSample] = []
        texts_set = set(
            Path(x.parent.name) / x.stem
            for x in self._text_dir.rglob(f'*{self._text_ext}')
        )
        mels_set = set(
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f'*{self._mels_ext}')
        )
        samples = list(mels_set & texts_set)
        for sample in tqdm(samples):
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            text_grid = tgt.read_textgrid(tg_path)

            if self.PHONES_TIER not in text_grid.get_tier_names():
                continue

            phones_tier = text_grid.get_tier_by_name(self.PHONES_TIER)

            phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]

            if self.LEXICON_OOV_TOKEN in phonemes:
                continue

            speakers_counter = self.add_to_mapping(
                self.speaker_to_id, sample.parent.name, speakers_counter
            )
            speaker_id = self.speaker_to_id[sample.parent.name]

            phoneme_ids = []
            for phoneme in phonemes:
                if phoneme == self.MFA_PAUSE_TOKEN:
                    phoneme = self.PAUSE_TOKEN
                phonemes_counter = self.add_to_mapping(
                    self.phoneme_to_id, phoneme, phonemes_counter
                )
                phoneme_ids.append(self.phoneme_to_id[phoneme])

            durations = [
                self.seconds_to_frame(x.duration())
                for x in phones_tier.get_copy_with_gaps_filled()
            ]

            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            mels: torch.Tensor = torch.load(mels_path)
            mels = (mels - MELS_MEAN) / MELS_STD

            pad_size = mels.shape[-1] - int(sum(durations))
            # assert pad_size >= 0, f'Expected {mels.shape[-1]} mel frames, got {sum(input_sample["durations"])}'
            # TODO: fix problem when pad_size < 0
            if pad_size < 0:
                durations[-1] -= pad_size
                assert durations[-1] >= 0
            if pad_size > 0:
                phoneme_ids.append(self.phoneme_to_id[self.PAUSE_TOKEN])
                durations.append(pad_size)

            dataset.append(
                VCTKSample(
                    phonemes=phoneme_ids,
                    num_phonemes=len(phoneme_ids),
                    speaker_id=speaker_id,
                    durations=durations,
                    mels=mels,
                )
            )
        return dataset

    def split_train_valid(
        self, test_fraction: float
    ) -> Tuple[VctkDataset, VctkDataset]:
        speakers_to_data_id: Dict[int, List[int]] = defaultdict(list)

        for i, sample in enumerate(self._dataset):
            speakers_to_data_id[sample.speaker_id].append(i)
        test_ids: List[int] = []
        for ids in speakers_to_data_id.values():
            test_size = int(len(ids) * test_fraction)
            if test_size > 0:
                test_indexes = random.choices(ids, k=test_size)
                test_ids.extend(test_indexes)

        train_data = []
        test_data = []
        for i in range(len(self._dataset)):
            if i in test_ids:
                test_data.append(self._dataset[i])
            else:
                train_data.append(self._dataset[i])
        return VctkDataset(train_data), VctkDataset(test_data)

    def save_mapping(self, path: Path):
        with open(path / self.SPEAKER_JSON_NAME, "w") as f:
            json.dump(self.speaker_to_id, f)
        with open(path / self.PHONEMES_JSON_NAME, "w") as f:
            json.dump(self.phoneme_to_id, f)


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
        batch_size = len(batch)
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x.phonemes) for x in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = input_lengths[0]

        input_speaker_ids = torch.LongTensor(
            [batch[i].speaker_id for i in ids_sorted_decreasing]
        )

        text_padded = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        durations_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx].phonemes
            text_padded[i, : len(text)] = torch.LongTensor(text)
            durations = batch[idx].durations
            durations_padded[i, : len(durations)] = torch.FloatTensor(durations)

        # Right zero-pad mel-spec
        num_mels = batch[0].mels.squeeze(0).size(0)
        max_target_len = max([x.mels.squeeze(0).size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.zeros(
            (batch_size, num_mels, max_target_len), dtype=torch.float
        )
        for i, idx in enumerate(ids_sorted_decreasing):
            mel = batch[idx].mels.squeeze(0)
            mel_padded[i, :, : mel.size(1)] = mel
        mel_padded = mel_padded.permute(0, 2, 1)
        return VCTKBatch(
            phonemes=text_padded,
            num_phonemes=input_lengths,
            speaker_ids=input_speaker_ids,
            durations=durations_padded,
            mels=mel_padded,
        )
