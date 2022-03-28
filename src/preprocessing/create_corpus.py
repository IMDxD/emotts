#!/usr/bin/env python
"""Util to create phonematic corpus for dataset.
You must use already pretrained G2P model for suitable language.
All dataset texts must be already extracted into text files.
"""


from pathlib import Path
from typing import List

import click
from text.cleaners import russian_cleaners
from tqdm import tqdm

ENCODING = "utf8"


@click.command()
@click.option(
    "--input-dir", type=Path, help="Directory with rearranged dataset.", required=True,
)
@click.option(
    "--output-path", type=Path, default="corpus.txt", help="Filepath to write corpus.", required=True,
)
@click.option(
    "--valid-extensions", type=str, multiple=True, default=["txt"], help="Extensions of text files.", required=True,
)
def main(
    input_dir: Path,
    output_path: Path,
    valid_extensions: List[str],
) -> None:

    output_path.parent.mkdir(exist_ok=True, parents=True)
    filepath_list = [fp for fp in input_dir.rglob("*") if fp.suffix[1:] in valid_extensions]
    print(f"{len(filepath_list)} text files found. Creating corpus...")

    corpus = set()
    for path in tqdm(filepath_list):
        with open(path, "r", encoding=ENCODING) as f:
            text = f.read()
            text = russian_cleaners(text)
            words = text.split()
            for word in words:
                corpus.add(word)
    
    with open(output_path, "w", encoding=ENCODING) as f:
        f.write("\n".join(sorted(corpus)))
    

if __name__ == "__main__":
    main()
