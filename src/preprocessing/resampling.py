#!/usr/bin/env python
"""Resamples audio and converts stereo to mono"""
from pathlib import Path

import click
from torchaudio import (
    info as torchaudio_info, load as torchaudio_load, save as torchaudio_save,
)
from torchaudio.transforms import Resample
from tqdm import tqdm


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory with audios to process.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory for audios with pauses trimmed.")
@click.option("--resample-rate", type=int, default=22050, required=True,
              help="Resulting sample rate in Hz.")
@click.option("--audio-ext", type=str, default="flac", required=True,
              help="Extension of audio files.")
def main(input_dir: Path, output_dir: Path, audio_ext: str, resample_rate: int) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    filepath_list = list(input_dir.rglob(f"*.{audio_ext}"))
    print(f"Number of audio files found: {len(filepath_list)}")

    sample_rate = torchaudio_info(filepath_list[0]).sample_rate
    resampler = Resample(
        orig_freq=sample_rate,
        new_freq=resample_rate,
        resampling_method="sinc_interpolation",
    )

    print(f"Resampling audio from {sample_rate} Hz to {resample_rate} Hz...")

    for filepath in tqdm(filepath_list):
        new_dir = output_dir / filepath.parent.name
        new_dir.mkdir(exist_ok=True)

        waveform, _ = torchaudio_load(filepath)

        new_waveform = resampler(waveform)

        # stereo to mono
        mono_waveform = new_waveform.mean(axis=0, keepdim=True)
        assert mono_waveform.shape[0] == 1, "Audio has more than 1 channel"

        torchaudio_save(new_dir / filepath.name, mono_waveform, resample_rate)

    print("Finished successfully.")
    print(f"Processed files are located at {output_dir}")


if __name__ == "__main__":
    main()
