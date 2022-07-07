import random
from pathlib import Path
from typing import List

from src.data_process.config import VCTKDatasetParams


def get_mel_file_path(full_wav_name: str, mels_dir: str, suffix: str = ".pth"):
    return Path(
        mels_dir, Path(*list(Path(full_wav_name).parts[-2:])).with_suffix(suffix)
    )


def split_vctk_data(data_config: VCTKDatasetParams, test_fraction: float):
    wavs_dir_path = Path(data_config.wav_dir)
    mels_dir_path = Path(data_config.feature_dir)

    audio_set = {x.stem for x in wavs_dir_path.rglob("*wav")}
    mels_set = {x.stem for x in mels_dir_path.rglob("*pth")}
    samples = list(mels_set & audio_set)

    speakers_to_data_id = {
        speaker.name: list(speaker.iterdir()) for speaker in mels_dir_path.iterdir()
    }
    test_ids: List[int] = []
    for speaker, ids in speakers_to_data_id.items():
        test_size = int(len(ids) * test_fraction)
        if test_size > 0 and speaker in data_config.ignore_speakers:
            test_indexes = random.choices([file.stem for file in ids], k=test_size)
            test_ids.extend(test_indexes)

    train_data = []
    test_data = []
    for file in wavs_dir_path.rglob("*wav"):
        if file.stem in samples:
            if file.stem in test_ids:
                test_data.append(file.as_posix())
            else:
                train_data.append(file.as_posix())

    return train_data, test_data
