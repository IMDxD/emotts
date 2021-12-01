#!/usr/bin/env python
import shutil
from pathlib import Path
from typing import List

import click
from tqdm import tqdm


VALID_EMOTIONS = ["angry", "happy", "sad", "veryangry", "veryhappy"]


@click.command()
@click.option("--input-dir", type=Path,)
@click.option("--output-dir", type=Path, default="mfa_outputs",)
@click.option("--log-path", type=Path, default="logs/preprocessing/mfa-premade-skipped.txt",)
@click.option("--ext", type=str, multiple=True, default=["TextGrid"],)
def main(
    input_dir: Path,
    output_dir: Path,
    log_path: Path,
    ext: List[str],
) -> None:

    output_dir.mkdir(exist_ok=True, parents=True)
    log_path.parent.mkdir(exist_ok=True, parents=True)
    log_path.unlink(missing_ok=True)
    separator_old = "_"
    separator_new = "-"
    neutral_emotion_label = "neutral"

    all_paths = list(input_dir.rglob("*"))
    for path in tqdm(all_paths):
        
        # If TextGrid, get new path and copy it
        if path.suffix[1:] in ext:
            chunk_name = path.name.split(separator_old)[-1]
            
            # emo case
            if separator_old in path.parent.name:
                emotion, speaker = path.parent.name.split(separator_old, 1)
                # esq_dots, wiki_len etc.
                if emotion not in VALID_EMOTIONS:
                    emotion = neutral_emotion_label
                    speaker = path.parent.name

            # neutral case
            else:
                emotion = neutral_emotion_label
                speaker = path.parent.name
            
            new_speaker = speaker.replace(separator_old, separator_new)
            new_dir = output_dir / emotion / new_speaker
            new_filename = separator_old.join((new_speaker, emotion, chunk_name))
            new_path = new_dir / new_filename
            new_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(path, new_path)

        # else do nothing, log skipped path
        else:
            with open(log_path, "a") as logfile:
                logfile.write(f"{path}\n")
            continue


if __name__ == "__main__":
    main()
