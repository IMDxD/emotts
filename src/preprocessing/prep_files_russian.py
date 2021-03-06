#!/usr/bin/env python
import shutil
from pathlib import Path
from typing import List

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
    df = df.iloc[:, :3]
    df.columns = ["resource", "number", "sentence"]
    for idx, row in df.iterrows():
        try:
            filename = row["number"]
            content = row["sentence"]
        except KeyError:
            print(row)
            raise
        new_filename = f"{speaker}_{emotion}_{filename}.{text_ext}"
        new_dir = text_output_dir / speaker
        new_dir.mkdir(parents=True, exist_ok=True)
        new_filepath = new_dir / new_filename
        with open(new_filepath, "w") as text_output_file:
            text_output_file.write(content)


@click.command()
@click.option(
    "--dataset-dir", type=Path, help="Directory with original russian dataset"
)
@click.option(
    "--text-output-dir",
    type=Path,
    default="texts",
    help="Directory for text files extracted from annotations.",
)
@click.option(
    "--audio-output-dir",
    type=Path,
    default="wavs",
    help="Directory for rearranged audio files.",
)
@click.option(
    "--log-path",
    type=Path,
    default="logs/preprocessing/russian-skipped-paths.txt",
    help="Path for logging list of skipped items.",
)
@click.option(
    "--annot-ext",
    type=str,
    multiple=True,
    default=["xls", "xlsx"],
    help="Extension of annotation files.",
)
@click.option("--audio-ext", type=str, default="wav", help="Extension of audio files.")
def main(
    dataset_dir: Path,
    text_output_dir: Path,
    audio_output_dir: Path,
    log_path: Path,
    audio_ext: str,
    annot_ext: List[str],
) -> None:

    text_output_dir.mkdir(exist_ok=True, parents=True)
    audio_output_dir.mkdir(exist_ok=True, parents=True)
    log_path.parent.mkdir(exist_ok=True, parents=True)
    log_path.unlink(missing_ok=True)

    for path in tqdm(dataset_dir.rglob("*")):

        # If audio, get speaker, get emotion and copy it with new name
        if path.suffix == f".{audio_ext}":
            process_audio(path, audio_output_dir)
        # If annotation, parse it and rearrange texts
        elif path.suffix[1:] in annot_ext:
            process_annotation(path, text_output_dir)
        # else do nothing, log skipped path
        else:
            with open(log_path, "a") as logfile:
                logfile.write(f"{path}\n")
            continue


if __name__ == "__main__":
    main()
