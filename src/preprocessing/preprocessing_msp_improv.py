#!/usr/bin/env python
import shutil
from collections import defaultdict
from pathlib import Path

import click
from tqdm import tqdm

DATA_SUBSET_MODES = ["R"]


@click.command()
@click.option("--input-text-dir", type=Path,
              help="Directory with texts to process.")
@click.option("--input-audio-dir", type=Path,
              help="Directory with audios to process.")
@click.option("--output-text-dir", type=Path, default="trimmed",
              help="Directory for reorganized texts.")
@click.option("--output-audio-dir", type=Path, default="trimmed",
              help="Directory for reorganized audios.")
@click.option("--text-ext", type=str, default="txt",
              help="Extension of text files.")
@click.option("--audio-ext", type=str, default="flac",
              help="Extension of audio files.")
def main(input_text_dir: Path, input_audio_dir: Path,
         output_text_dir: Path, output_audio_dir: Path,
         text_ext: str, audio_ext: str) -> None:
    output_text_dir.mkdir(exist_ok=True, parents=True)
    output_audio_dir.mkdir(exist_ok=True, parents=True)

    audio_filepath_list = list(input_audio_dir.rglob(f"*.{audio_ext}"))

    print(f"Sifting {len(audio_filepath_list)} files from {input_audio_dir} to {output_audio_dir}...")

    name_prefixes = defaultdict(int)
    for audio_filepath in tqdm(audio_filepath_list):
        try:
            _, _, sent_emo, speaker_id, subset_mode, _ = audio_filepath.stem.split("-")  # ["MSP", "IMPROV", "S12H", "F02", "T", "FM01"]
        except ValueError:
            print(f"{audio_filepath}")

        corresponding_text_filepath = (input_text_dir / audio_filepath.stem).with_suffix(f".{text_ext}")
        if not corresponding_text_filepath.exists():
            continue

        if subset_mode not in DATA_SUBSET_MODES:
            continue

        name_prefix = speaker_id + "_" + sent_emo[:-1] + "_" + sent_emo[-1] + "_"  # "F02_S12_H_"
        new_name = name_prefix + f"{name_prefixes[name_prefix]:03d}"  # "F02_S12_H_000"
        name_prefixes[name_prefix] += 1

        # text
        new_dir = output_text_dir / speaker_id
        new_dir.mkdir(exist_ok=True)
        new_path = (new_dir / new_name).with_suffix(f".{text_ext}")
        shutil.copy(corresponding_text_filepath, new_path)

        # audio
        new_dir = output_audio_dir / speaker_id
        new_dir.mkdir(exist_ok=True)
        new_path = (new_dir / new_name).with_suffix(f".{audio_ext}")
        shutil.copy(audio_filepath, new_path)

    print(f"Resulting number of files in {output_audio_dir}: "
          f"{len(list(output_audio_dir.rglob(f'*.{audio_ext}')))}.")


if __name__ == "__main__":
    main()
