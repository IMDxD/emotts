#/usr/bin/env python
from pathlib import Path

import click
from librosa.filters import mel as librosa_mel
import torch
import torchaudio
from tqdm import tqdm


# Using the same parameters as in HiFiGAN
F_MIN = 0
F_MAX = 8000
HOP_SIZE = 256
WIN_SIZE = 1024
N_FFT = 1024
N_MELS = 80
SAMPLE_RATE = 22050


mel_basis = {}
hann_window = {}


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = torch.log(torch.clamp(magnitudes, min=1e-5))
    return output


def mel_spectrogram(y: torch.Tensor,
                    n_fft: int = N_FFT, num_mels: int = N_MELS,
                    sample_rate: int= SAMPLE_RATE, hop_size: int = HOP_SIZE,
                    win_size: int = WIN_SIZE, fmin: int = F_MIN, fmax: int = F_MAX,
                    center: bool = False) -> torch.Tensor:

    global mel_basis, hann_window

    if fmax not in mel_basis:
        mel = librosa_mel(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[f"{fmax}_{y.device}"] = torch.from_numpy(mel).float().to(y.device)
        hann_window[f"{y.device}"] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1),
                                (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)),
                                mode='reflect')

    spec = torch.stft(y.squeeze(1), n_fft, hop_length=hop_size,
                      win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect',
                      normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis[f"{fmax}_{y.device}"], spec)
    spec = spectral_normalize_torch(spec)

    return spec


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

    for file in tqdm(filepath_list):
        new_path = processed_path / file.stem

        wave_tensor, _ = torchaudio.load(file)

        mels_tensor = mel_spectrogram(wave_tensor)  # [n_channels x n_mels x time]
        torch.save(mels_tensor, new_path.with_suffix('.pkl'))

    print('Finished successfully.')
    print(f'Processed files are located at {output_dir}')


if __name__ == '__main__':
    main()
