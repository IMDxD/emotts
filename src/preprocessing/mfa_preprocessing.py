"""Flatten MFA files by speakers"""
from pathlib import Path
from shutil import move

import click
from tqdm import tqdm


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory to move audio from.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory to move audio to.")
def main(input_dir: Path, output_dir: Path) -> None:
    for _i, dir_path in tqdm(enumerate(input_dir.iterdir())):
        for _j, filepath in enumerate(dir_path.iterdir()):
            move(str(filepath), output_dir / dir_path.name)
        dir_path.rmdir()

    print(f"{_i + _j + 2} files were moved to {output_dir}")


if __name__ == "__main__":
    main()
