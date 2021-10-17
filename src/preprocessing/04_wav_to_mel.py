#/usr/bin/env python
from pathlib import Path

import click
from torch import save as torch_save
from torchaudio import load as torchaudio_load
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm


# Based on https://github.com/Illumaria/made-emotts-2021/blob/non-attentive-tacotron/default_params.py
F_MIN = 55
F_MAX = 7600
HOP_SIZE = 256
N_FFT = 1024
N_MELS = 80  # required by HiFi-GAN
NORMALIZED = True
SAMPLE_RATE = 22050
WIN_SIZE = 1024


@click.command()
@click.option('--input-dir', type=str, required=True,
              help='Directory with audios to process.')
@click.option('--output-dir', type=str, required=True,
              help='Directory for audios with pauses trimmed.')
def main(input_dir: str, output_dir: str):
    path = Path(input_dir)
    processed_path = Path(output_dir)
    processed_path.mkdir(exist_ok=True, parents=True)

    filepath_list = list(path.rglob('*.flac'))
    print(f'Number of audio files found: {len(filepath_list)}')
    print('Transforming audio to mel...')

    # TODO: determine transformation parameters
    transformer = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_SIZE,
        hop_length=HOP_SIZE,
        f_min=F_MIN,
        f_max=F_MAX,
        n_mels=N_MELS,
        normalized=NORMALIZED,
        # norm = 'slaney',
    )

    for file in tqdm(filepath_list):
        # new_dir = processed_path / file.parent.name
        # new_dir.mkdir(exist_ok=True)
        new_path = processed_path / file.stem

        wave_tensor, _ = torchaudio_load(file)

        new_tensor = transformer(wave_tensor)  # [n_channels x n_mels x time]
        # torch_save(new_tensor, (new_dir / file.stem).with_suffix('.pkl'))
        torch_save(new_tensor, new_path.with_suffix('.pkl'))

    print('Finished successfully.')
    print(f'Processed files are located at {output_dir}')

if __name__ == '__main__':
    main()
