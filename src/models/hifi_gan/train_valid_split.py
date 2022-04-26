import random
from pathlib import Path
from typing import List

from src.data_process.config import VCTKDatasetParams


def split_vctk_data(data_config: VCTKDatasetParams, test_size: float):
    wavs_dir_path = Path(data_config.wav_dir)
    mels_dir_path = Path(data_config.mels_dir)

    texts_set = {
        x.stem for x in wavs_dir_path.rglob("*wav")
    }
    mels_set = {
        x.stem for x in mels_dir_path.rglob(f"*{data_config.mels_ext}")
    }
    samples = list(mels_set & texts_set)

    speakers_to_data_id = {speaker: list(speaker.iterdir()) for speaker in mels_dir_path.iterdir()}
    test_ids: List[int] = []
    for speaker, ids in speakers_to_data_id.items():
        test_size = int(len(ids) * test_size)
        if test_size > 0 and speaker not in data_config.ignore_speakers:
            test_indexes = [file.stem for file in random.choices(ids, k=test_size)]
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
