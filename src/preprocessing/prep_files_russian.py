#!/usr/bin/env python
import shutil
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


def process_audio(audio_path: Path, audio_output_dir: Path) -> None:
    speaker = audio_path.parent.name.replace("_", "-")
    emotion = audio_path.parent.parent.name
    new_dir = audio_output_dir / speaker
    new_dir.mkdir(parents=True, exist_ok=True)
    new_filename = f"{speaker}_{emotion}_{audio_path.name}"
    new_audio_path = new_dir / new_filename
    shutil.copy(audio_path, new_audio_path)


def process_annotation(annot_path: Path, text_output_dir: Path) -> None:
    text_ext = "txt"
    speaker = annot_path.parent.name.replace("_", "-")
    emotion = annot_path.parent.parent.name
    df: pd.DataFrame = pd.read_excel(annot_path)
    for idx, row in df.iterrows():
        filename = row["number"]
        content = row["sentence"]
        new_filename = f"{speaker}_{emotion}_{filename}.{text_ext}"
        new_dir = text_output_dir / speaker
        new_dir.mkdir(parents=True, exist_ok=True)
        new_filepath = new_dir / new_filename
        with open(new_filepath, "w") as text_output_file:
            text_output_file.write(content)


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

    for path in tqdm(dataset_dir.rglob("*")):

        # If audio, get speaker, get emotion and copy it with new name
        if path.suffix == f".{audio_ext}":
            process_audio(path, audio_output_dir)

        # If annotation, parse it and rearrange texts
        elif path.suffix == f".{annot_ext}":
            process_annotation(path, text_output_dir)

        # else WTF
        else:
            print(f"Skipped: {path}")
            continue


if __name__ == "__main__":
    main()
