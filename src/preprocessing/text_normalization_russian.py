#!/usr/bin/env python
from pathlib import Path

import click
from tqdm import tqdm

from src.preprocessing.text.russian_stt_text_normalization.normalizer import Normalizer


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory with texts to process.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory for normalized texts.")
def main(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    normalizer = Normalizer()
    filepath_list = list(input_dir.rglob("*.txt"))
    print(f"Number of text files found: {len(filepath_list)}")
    print("Normalizing texts...")

    for filepath in tqdm(filepath_list):
        new_dir = output_dir / filepath.parent.name
        new_dir.mkdir(exist_ok=True)
        new_file = new_dir / filepath.name

        with open(filepath, "r") as fin, open(new_file, "w") as fout:
            content = fin.read()
            normalized_content = normalizer.norm_text(content)
            fout.write(normalized_content)

    print("Finished successfully.")
    print(f"Processed files are located at {output_dir}")


if __name__ == "__main__":
    main()
