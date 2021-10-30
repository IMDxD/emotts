"""Flatten MFA files by speakers"""
import click
from pathlib import Path
from shutil import move


@click.command()
@click.option('--input-dir', type=str, required=True,
              help='Directory to move audio from.')
@click.option('--output-dir', type=str, required=True,
              help='Directory to move audio to.')
def main(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for dir_path in input_dir.iterdir():
        for file in dir_path.iterdir():
            move(str(file), output_dir / dir_path.name)
        dir_path.rmdir()


if __name__ == "__main__":
    main()
