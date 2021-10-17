from pathlib import Path

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
        self._speaker_ids = sorted(f.name for f in self._text_dir.iterdir() if f.is_dir())
        self._sample_ids = []

        wrong_speaker_ids = []
        for speaker_id in self._speaker_ids:
            utterance_dir = self._text_dir / speaker_id
            speaker_sample_ids = []
            for utterance_file in sorted(f for f in utterance_dir.iterdir() if f.suffix == self._text_ext):
                utterance_id = utterance_file.stem
                audio_path = (self._audio_dir / utterance_id).with_suffix(self._audio_ext)
                if audio_path.exists():
                    speaker_sample_ids.append(utterance_id.split("_"))
            if len(speaker_sample_ids) == 0:
                wrong_speaker_ids.append(speaker_id)
            else:
                self._sample_ids.extend(speaker_sample_ids)

        # TODO: IF there are any, remove them from self._speaker_ids
        if len(wrong_speaker_ids) > 0:
            print(f'Speaker IDs with no audio: {wrong_speaker_ids}.')
        
        print(f'Found {len(self._sample_ids)} samples for {len(self._speaker_ids)} speakers.')
