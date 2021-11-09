from pathlib import Path, PurePath
import numpy as np

NON_VALID_SPEAKERS = ["p315"]
VALID_SIZE = 9


def get_mel_file_path(full_wav_name: str, mels_dir: str):
    return Path(
            mels_dir,
            Path(*list(Path(full_wav_name).parts[1:])).with_suffix(".pth")
            )


def split_vctk_data(wavs_dir: str, mels_dir: str):
    wavs_dir_path = Path(wavs_dir)
    speaker_ids = []
    for dir in wavs_dir_path.iterdir():
        if dir.is_dir() and dir.name not in NON_VALID_SPEAKERS:
            speaker_ids.append(dir.name)

    np.random.shuffle(speaker_ids)
    train_ids, valid_ids = speaker_ids[:- VALID_SIZE], speaker_ids[- VALID_SIZE:]
    training_files, validation_files = [], []
    for dir in wavs_dir_path.iterdir():
        if dir.name in speaker_ids:
            for file in dir.iterdir():
                full_wav_name = PurePath(file).as_posix()
                mels_file_path = get_mel_file_path(full_wav_name, mels_dir)
                if mels_file_path.is_file():
                    if dir.name in train_ids:
                        training_files.append(full_wav_name)
                    else:
                        validation_files.append(full_wav_name)

    return training_files, validation_files
