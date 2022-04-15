#!/usr/bin/env python
from pathlib import Path

import click
from text.cleaners import collapse_whitespace, lowercase
from text.russian.normalizer import Normalizer
from tqdm import tqdm

NORMALIZER_MODEL_PATH = "src/preprocessing/text/russian/jit_s2s.pt"


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory with texts to process.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory for normalized texts.")
def main(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    normalizer = Normalizer(jit_model=NORMALIZER_MODEL_PATH)
    filepath_list = list(input_dir.rglob("*.txt"))
    print(f"Number of text files found: {len(filepath_list)}")
    print("Normalizing texts...")

    for filepath in tqdm(filepath_list):
        new_dir = output_dir / filepath.parent.name
        new_dir.mkdir(exist_ok=True)
        new_file = new_dir / filepath.name

        with open(filepath, "r", encoding="utf8") as fin:
            with open(new_file, "w", encoding="utf8") as fout:
                content = fin.read()
                normalized_content = normalizer.norm_text(content)
                normalized_content = lowercase(normalized_content)
                normalized_content = collapse_whitespace(normalized_content)
                fout.write(normalized_content)

    print("Finished successfully.")
    print(f"Processed files are located at {output_dir}")


if __name__ == "__main__":
    main()
