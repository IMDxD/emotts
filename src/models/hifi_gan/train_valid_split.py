from pathlib import Path, PurePath
import numpy as np

NON_VALID_SPEAKERS = ["p315"]
VALID_SIZE = 9


def split_vctk_data(wavs_dir: str):
    wavs_dir_path = Path(wavs_dir)
    speaker_ids = []
    for dir in wavs_dir_path.iterdir():
        if dir.is_dir() and dir.name not in NON_VALID_SPEAKERS:
            speaker_ids.append(dir.name)

    np.random.shuffle(speaker_ids)
    train_ids, valid_ids = speaker_ids[:- VALID_SIZE], speaker_ids[- VALID_SIZE:]
    training_files, validation_files = [], []
    for dir in wavs_dir_path.iterdir():
        if dir.name in train_ids:
            for file in dir.iterdir():
                training_files.append(PurePath(file).as_posix())
        elif dir.name in valid_ids:
            for file in dir.iterdir():
                validation_files.append(PurePath(file).as_posix())

    return training_files, validation_files
