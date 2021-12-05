"""Flatten MFA files by speakers"""
from pathlib import Path
from shutil import copy

import click
from tqdm import tqdm


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory to move audio from.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory to move audio to.")
def main(input_dir: Path, output_dir: Path) -> None:
    files_total = 0
    for dir_path in tqdm(input_dir.iterdir()):
        new_dir_path = output_dir / dir_path.name
        new_dir_path.mkdir(exist_ok=True, parents=True)
        for filepath in dir_path.iterdir():
            copy(str(filepath), new_dir_path / filepath.name)
            files_total += 1

    print(f"{files_total} files were copied to {output_dir}")


if __name__ == "__main__":
    main()
