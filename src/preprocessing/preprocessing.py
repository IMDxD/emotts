#!/usr/bin/env python
import shutil
from pathlib import Path

import click
from tqdm import tqdm

PATTERN = "_mic"


@click.command()
@click.option("--input-dir", type=Path,
              help="Directory with audios to process.")
@click.option("--output-dir", type=Path, default="trimmed",
              help="Directory for audios with pauses trimmed.")
@click.option("--audio-ext", type=str, default="flac",
              help="Extension of audio files.")
def main(input_dir: Path, output_dir: Path, audio_ext: str) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    filepath_list = list(input_dir.rglob(f"*.{audio_ext}"))
    filepath_list = sorted(filepath_list, key=lambda x: x.name, reverse=True)

    print(f"Sifting {len(filepath_list)} files from {input_dir} to {output_dir}...")

    for filepath in tqdm(filepath_list):
        new_name = filepath.stem.split(PATTERN)[0] + filepath.suffix
        new_dir = output_dir / filepath.parent.name
        new_dir.mkdir(exist_ok=True)
        new_path = new_dir / new_name
        if PATTERN in filepath.name and not new_path.exists():
            shutil.copy(filepath, new_path)

    print(f"Resulting number of files in {output_dir}: "
          f"{len(list(output_dir.rglob(f'*.{audio_ext}')))}.")


if __name__ == "__main__":
    main()
