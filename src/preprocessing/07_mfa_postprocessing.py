"""Aggregate MFA files by speakers"""
import click
from pathlib import Path
from shutil import move


@click.command()
@click.option('--input-dir', type=str, required=True,
              help='Directory with mels to process.')
def main(input_dir: str):
    input_dir = Path(input_dir)

    for file in input_dir.iterdir():
        file_name = file.name
        dir_name = file_name.split('_')[0]
        dir_path = file.parent / dir_name
        dir_path.mkdir(exist_ok=True)
        move(str(file), dir_path)


if __name__ == "__main__":
    main()
