#/usr/bin/env python
from pathlib import Path

import click
from torchaudio import (
    info as torchaudio_info,
    load as torchaudio_load,
    save as torchaudio_save,
)
from torchaudio.transforms import Resample
from tqdm import tqdm


@click.command()
@click.option('--input-dir', type=str, required=True,
              help='Directory with audios to process.')
@click.option('--output-dir', type=str, required=True,
              help='Directory for audios with pauses trimmed.')
@click.option('--resample-rate', type=int, default=22050, required=True,
              help='Resulting sample rate in Hz.')
def main(input_dir: str, output_dir: str, resample_rate: int):
    path = Path(input_dir)
    processed_path = Path(output_dir)
    processed_path.mkdir(exist_ok=True, parents=True)

    filepath_list = list(path.rglob('*.flac'))
    print(f'Number of audio files found: {len(filepath_list)}')

    sample_rate = torchaudio_info(filepath_list[0]).sample_rate
    resampler = Resample(
        orig_freq=sample_rate,
        new_freq=resample_rate,
        resampling_method='sinc_interpolation',
    )

    print(f'Resampling audio from {sample_rate} Hz to {resample_rate} Hz...')

    for file in tqdm(filepath_list):
        new_dir = processed_path / file.parent.name
        new_dir.mkdir(exist_ok=True)

        waveform, _ = torchaudio_load(file)

        new_waveform = resampler(waveform)

        torchaudio_save(new_dir / file.name, new_waveform, resample_rate)

    print('Finished successfully.')
    print(f'Processed files are located at {output_dir}')

if __name__ == '__main__':
    main()
