from pathlib import Path

import tgt
from torch import load as torch_load
from torch.utils.data import Dataset


class VCTK(Dataset):
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
        # self._speaker_ids = list(set(x.parent.name for x in self._samples))
        # print(f'Found {len(self._samples)} samples for {len(self._speaker_ids)} speakers.')
        print(f'Found {len(self._samples)} samples.')
        print(f'Number of broken samples: {len(broken_samples)}.')

        self._dataset = []
        for sample in self._samples:
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            text_grid = tgt.read_textgrid(tg_path)

            assert 'phones' in text_grid.get_tier_names()
            phones_tier = text_grid.get_tier_by_name('phones')

            input_sample = {}
            input_sample['num_phonemes'] = len(phones_tier.intervals)
            input_sample['speaker_id'] = sample.parent.name
            input_sample['phonemes'] = [interval.text for interval in phones_tier.intervals]
            input_sample['durations'] = [interval.duration() for interval in phones_tier.intervals]

            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            mels = torch_load(mels_path)
            input_sample['mels'] = mels

            self._dataset.append(input_sample)

        # In DataLoader, we want to put in batch samples
        # with close num_phonemes values ([i:i + batch_size]),
        # so we sort dataset here and never shuffle it afterwards:
        self._dataset.sort(key=lambda x: x['num_phonemes'])

    def __len__(self):
        return len(self._sample_ids)

    def __getitem__(self, idx):
        return self._dataset[idx]
