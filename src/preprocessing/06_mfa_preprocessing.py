"""Flatten MFA files by speakers"""
from pathlib import Path
from shutil import move

import click
from tqdm import tqdm


@click.command()
@click.option('--input-dir', type=str, required=True,
              help='Directory to move audio from.')
@click.option('--output-dir', type=str, required=True,
              help='Directory to move audio to.')
def main(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    file_cnt = 0
    for dir_path in tqdm(input_dir.iterdir()):
        for file in dir_path.iterdir():
            move(str(file), output_dir / dir_path.name)
            file_cnt += 1
        dir_path.rmdir()

    print(f"{file_cnt} files were moved to {output_dir}")


if __name__ == "__main__":
    main()
