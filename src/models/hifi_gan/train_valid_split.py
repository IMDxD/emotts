from pathlib import Path, PurePath

import numpy as np

NON_VALID_SPEAKERS = ["p315"]
VALID_SIZE = 9
SPLIT_LOG_PATH = Path("logs/hifi-split.txt")


def get_mel_file_path(full_wav_name: str, mels_dir: str):
    return Path(
            mels_dir,
            Path(*list(Path(full_wav_name).parts[-2:])).with_suffix(".pth")
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

    # Save  train and val filenames for sanity check
    Path(*list(SPLIT_LOG_PATH.parts[:-1])).mkdir(parents=True, exist_ok=True)
    with open(SPLIT_LOG_PATH, "w") as log_file:
        log_file.write("train:")
        log_file.writelines(training_files)
        log_file.write("val:")
        log_file.writelines(validation_files)

    return training_files, validation_files
