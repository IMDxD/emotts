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
    esd_speakers = [folder.name for folder in wavs_dir.iterdir() if folder.name.startswith("00")]
    for folder in wavs_dir_path.iterdir():
        if folder.is_dir() and folder.name not in NON_VALID_SPEAKERS and not folder.name.startswith("00"):
            speaker_ids.append(folder.name)

    np.random.shuffle(speaker_ids)
    train_ids, valid_ids = speaker_ids[:-VALID_SIZE], speaker_ids[-VALID_SIZE:]
    train_ids.extend(esd_speakers)
    print(f"Train speakers: {train_ids}")
    print(f"Valid speakers: {valid_ids}")
    training_files, validation_files = [], []
    for folder in wavs_dir_path.iterdir():
        if folder.name in speaker_ids:
            for file in folder.iterdir():
                full_wav_name = PurePath(file).as_posix()
                mels_file_path = get_mel_file_path(full_wav_name, mels_dir)
                if mels_file_path.is_file():
                    if folder.name in train_ids:
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
