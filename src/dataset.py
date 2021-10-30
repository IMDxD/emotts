from pathlib import Path

import tgt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.preprocessing.text.cmudict import valid_symbols

HOP_SIZE = 256
PAUSE_TOKEN = "<SIL>"
LEXICON_OOV_TOKEN = "spn"
SAMPLE_RATE = 22050
PHONEME_TO_IDX = {
    phoneme: idx
    for idx, phoneme in enumerate([PAUSE_TOKEN] + valid_symbols)
}

class VctkDataset(Dataset):
    """Create VCTK Dataset

    Args:
        root (str): Root directory where the dataset's top level directory is found.
        mic_id (str): Microphone ID. Either ``"mic1"`` or ``"mic2"``. (default: ``"mic2"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"``)
        audio_ext (str, optional): Custom audio extension if dataset is converted to non-default audio format.

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
        text_dir: str,
        mels_dir: str,
        text_ext: str = ".TextGrid",
        mels_ext: str = ".pkl",
    ):
        self._text_dir = Path(text_dir)
        self._mels_dir = Path(mels_dir)
        self._text_ext = text_ext
        self._mels_ext = mels_ext

        # Check that input dirs exist:
        if not self._text_dir.is_dir():
            raise FileNotFoundError(f'Text data not found at {self._text_dir}.')
        if not self._mels_dir.is_dir():
            raise FileNotFoundError(f'Mels data not found at {self._mels_dir}.')

        # Extracting speaker IDs from the folder structure
        texts = set(Path(x.parent.name) / x.stem
                    for x in self._text_dir.rglob(f'*{self._text_ext}'))
        mels = set(Path(x.parent.name) / x.stem
                   for x in self._mels_dir.rglob(f'*{self._mels_ext}'))

        self._samples = list(texts & mels)
        broken_samples = texts - mels
        print(f'Number of broken samples: {len(broken_samples)}.')

        self._dataset = []
        self._build_dataset()
        print(f'Found {len(self._dataset)} samples.')

    def _build_dataset(self):
        for sample in tqdm(self._samples):
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            text_grid = tgt.read_textgrid(tg_path)

            if 'phones' not in text_grid.get_tier_names():
                continue

            phones_tier = text_grid.get_tier_by_name('phones')

            input_sample = {}
            input_sample['num_phonemes'] = len(phones_tier.intervals)
            input_sample['speaker_id'] = sample.parent.name

            if LEXICON_OOV_TOKEN in [x.text for x in phones_tier.get_copy_with_gaps_filled()]:
                continue

            input_sample['phonemes'] = [PHONEME_TO_IDX[x.text] if x.text else PHONEME_TO_IDX[PAUSE_TOKEN]
                                        for x in phones_tier.get_copy_with_gaps_filled()]
            input_sample['durations'] = [x.duration() * SAMPLE_RATE / HOP_SIZE
                                         for x in phones_tier.get_copy_with_gaps_filled()]

            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            mels = torch.load(mels_path)
            input_sample['mels'] = mels

            pad_size = mels.shape[-1] - sum(input_sample['durations'])
            # assert pad_size >= 0, f'Expected {mels.shape[-1]} mel frames, got {sum(input_sample["durations"])}'
            # TODO: fix problem when pad_size < 0
            if pad_size < 0:
                # print(f"Removing {-pad_size} frames from input sample duration.")
                input_sample['durations'][-1] -= pad_size
                assert input_sample['durations'][-1] >= 0
            if pad_size > 0:
                input_sample['phonemes'].append(PHONEME_TO_IDX[PAUSE_TOKEN])
                input_sample['durations'].append(pad_size)

            self._dataset.append(input_sample)

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

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [{}, {}, ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x["phonemes"]) for x in batch]),
            dim=0, descending=True,
        )
        max_input_len = input_lengths[0]

        text_padded = torch.zeros((len(batch), max_input_len), dtype=torch.long)
        durations_padded = torch.zeros((len(batch), max_input_len), dtype=torch.long)
        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx]["phonemes"]
            text_padded[i, :len(text)] = torch.as_tensor(text)
            durations = batch[idx]["durations"]
            durations_padded[i, :len(durations)] = torch.as_tensor(durations)

        # Right zero-pad mel-spec
        num_mels = batch[0]["mels"].squeeze(0).size(0)
        max_target_len = max([x["mels"].squeeze(0).size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i, idx in enumerate(ids_sorted_decreasing):
            mel = batch[idx]["mels"].squeeze(0)
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
