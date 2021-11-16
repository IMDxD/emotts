#!/usr/bin/env python
import shutil
from pathlib import Path

import click
from tqdm import tqdm

PATTERN = "_mic"


@click.command()
@click.option("--dataset-dir", type=Path,
              help="Directory with russian dataset")
@click.option("--text-output-dir", type=Path, default="trimmed",
              help="Directory for audios with pauses trimmed.")
@click.option("--audio-output-dir", type=Path, default="trimmed",
              help="Directory for audios with pauses trimmed.")
@click.option("--annot-ext", type=str, default="xlsx",
              help="Extension of audio files.")
@click.option("--audio-ext", type=str, default="wav",
              help="Extension of audio files.")
def main(dataset_dir: Path, text_output_dir: Path, audio_output_dir: Path, audio_ext: str, annot_ext: str) -> None:
    text_output_dir.mkdir(exist_ok=True, parents=True)
    audio_output_dir.mkdir(exist_ok=True, parents=True)

    audio_list = list(dataset_dir.rglob(f"*.{audio_ext}"))
    annot_list = list(dataset_dir.rglob(f"*.{annot_ext}"))

    print(*audio_list, sep="\n")
    print()
    print(*annot_list, sep="\n")

    # print(f"Sifting {len(filepath_list)} files from {dataset_dir} to {text_output_dir}...")

    # for filepath in tqdm(filepath_list):
    #     new_name = filepath.stem.split(PATTERN)[0] + filepath.suffix
    #     new_dir = text_output_dir / filepath.parent.name
    #     new_dir.mkdir(exist_ok=True)
    #     new_path = new_dir / new_name
    #     if PATTERN in filepath.name and not new_path.exists():
    #         shutil.copy(filepath, new_path)

    # print(f"Resulting number of files in {text_output_dir}: "
    #       f"{len(list(text_output_dir.rglob(f'*.{audio_ext}')))}.")

    # RUSSIAN_DATASET_PATH=/media/diskB/ruslan_a/data/datasets/EMO/russian/

if __name__ == "__main__":
    main()