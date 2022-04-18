#!/usr/bin/env python
import json
import shutil
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


def process_audio(audio_path: Path, audio_output_dir: Path, emo_speaker: defaultdict) -> None:
    old_dir = audio_path.parent.name.replace("_", "-")
    speaker = "olga"
    emotion = "neutral"
    new_dir = audio_output_dir / speaker
    new_dir.mkdir(parents=True, exist_ok=True)
    new_filename = f"{old_dir}-{emotion}_{audio_path.name}"
    new_audio_path = new_dir / new_filename
    emo_speaker[emotion][speaker].add(new_audio_path.stem)
    shutil.copy(audio_path, new_audio_path)


def process_metadata(metadata_path: Path, text_output_dir: Path, emo_speaker: defaultdict) -> None:
    text_ext = "txt"
    speaker = "olga"
    emotion = "neutral"
    df: pd.DataFrame = pd.read_csv(metadata_path, delimiter="|", header=None, names=["path", "original", "stressed"])
    for (_, path, original, stressed) in df.itertuples():
        old_dir, filename = path.split("/")
        new_filename = f"{old_dir}-{emotion}_{filename}.{text_ext}"
        new_dir = text_output_dir / speaker
        new_dir.mkdir(parents=True, exist_ok=True)
        new_filepath = new_dir / new_filename
        emo_speaker[emotion][speaker].add(new_filepath.stem)
        with open(new_filepath, "w", encoding="utf8") as text_output_file:
            text_output_file.write(original)


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
    "--meta-output-dir",
    type=Path,
    default="meta",
    help="Directory for newly created metadata files.",
)
@click.option(
    "--log-path",
    type=Path,
    default="logs/preprocessing/russian-skipped-paths.txt",
    help="Path for logging list of skipped items.",
)
@click.option(
    "--metadata-filename",
    type=str,
    default="metadata.csv",
    help="Name of metadata file to search in dataset.",
)
@click.option("--audio-ext", type=str, default="wav", help="Extension of audio files.")
def main(
    dataset_dir: Path,
    text_output_dir: Path,
    audio_output_dir: Path,
    meta_output_dir: Path,
    log_path: Path,
    audio_ext: str,
    metadata_filename: str,
) -> None:

    text_output_dir.mkdir(exist_ok=True, parents=True)
    audio_output_dir.mkdir(exist_ok=True, parents=True)
    meta_output_dir.mkdir(exist_ok=True, parents=True)
    log_path.parent.mkdir(exist_ok=True, parents=True)
    log_path.unlink(missing_ok=True)

    emo_speaker_json = defaultdict(lambda: defaultdict(set))

    for path in tqdm(list(dataset_dir.rglob("*"))):
        # If audio, get speaker, get emotion and copy it with new name
        if path.suffix == f".{audio_ext}":
            process_audio(path, audio_output_dir, emo_speaker_json)
        # If metadata, parse it and rearrange texts
        elif path.name == metadata_filename:
            process_metadata(path, text_output_dir, emo_speaker_json)
        # else do nothing, log skipped path
        else:
            with open(log_path, "a") as logfile:
                logfile.write(f"{path}\n")
            continue

    emo_speaker_json = dict(emo_speaker_json)
    for emotion in emo_speaker_json.keys():
        for speaker in emo_speaker_json[emotion].keys():
            emo_speaker_json[emotion][speaker] = list(emo_speaker_json[emotion][speaker])
    with open(meta_output_dir / "neutral_speaker_file.json", "w", encoding="utf8") as f:
        json.dump(emo_speaker_json, f)


if __name__ == "__main__":
    main()
