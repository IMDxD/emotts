"""Aggregate MFA files by speakers"""
from pathlib import Path
from shutil import move

import click


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory with mels to process.")
def main(input_dir: Path) -> None:
    for filepath in input_dir.iterdir():
        file_name = filepath.name
        dir_name = file_name.split("_")[0]
        dir_path = filepath.parent / dir_name
        dir_path.mkdir(exist_ok=True)
        move(str(filepath), dir_path)


if __name__ == "__main__":
    main()
