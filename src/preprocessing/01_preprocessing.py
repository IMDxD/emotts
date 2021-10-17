#/usr/bin/env python
import shutil
from pathlib import Path

import click
from tqdm import tqdm


PATTERN = '_mic'


@click.command()
@click.option('--input-dir', type=str,
              help='Directory with audios to process.')
@click.option('--output-dir', type=str, default='trimmed',
              help='Directory for audios with pauses trimmed.')
@click.option('--audio-ext', type=str, default='flac',
              help='Extension of audio files.')
def main(input_dir: str, output_dir: str, audio_ext: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    filepath_list = list(input_dir.rglob(f'*.{audio_ext}'))
    filepath_list = sorted(filepath_list, key=lambda x: x.name, reverse=True)

    print(f'Sifting {len(filepath_list)} files from {input_dir} to {output_dir}...')

    for file in tqdm(filepath_list):
        new_name = file.stem.split(PATTERN)[0] + file.suffix
        new_dir = output_dir / file.parent.name
        new_dir.mkdir(exist_ok=True)
        new_path = new_dir / new_name
        if PATTERN in file.name:
            if not new_path.exists():
                shutil.copy(file, new_path)

    print(f'Resulting number of files in {output_dir}: ' \
          f'{len(list(output_dir.rglob(f"*.{audio_ext}")))}.')

if __name__ == '__main__':
    main()
