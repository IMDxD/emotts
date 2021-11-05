#!/usr/bin/env python
from pathlib import Path

import click
from text.cleaners import english_cleaners
from tqdm import tqdm


@click.command()
@click.option('--input-dir', type=Path, required=True,
              help='Directory with texts to process.')
@click.option('--output-dir', type=Path, required=True,
              help='Directory for normalized texts.')
def main(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    filepath_list = list(input_dir.rglob('*.txt'))
    print(f'Number of text files found: {len(filepath_list)}')
    print('Normalizing texts...')

    for file in tqdm(filepath_list):
        new_dir = output_dir / file.parent.name
        new_dir.mkdir(exist_ok=True)
        new_file = new_dir / file.name
        # new_file = processed_path / file.name

        with open(file, 'r') as fin, open(new_file, 'w') as fout:
            content = fin.read()
            normalized_content = english_cleaners(content)
            fout.write(normalized_content)

    print('Finished successfully.')
    print(f'Processed files are located at {output_dir}')


if __name__ == '__main__':
    main()
