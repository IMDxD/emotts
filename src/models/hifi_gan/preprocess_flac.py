from os import listdir
from os.path import isfile, join
from pathlib import PurePath

import click
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import read, write
from scipy.signal import resample


def downsample(file_path: str, new_sample_rate: int, output_dir: str) -> None:
    sample_rate, waveform = read(file_path)

    if sample_rate != new_sample_rate:
        number_of_samples = round(waveform.shape[0] * float(new_sample_rate) / sample_rate)
        new_waveform = resample(waveform, number_of_samples)
        file_name = PurePath(file_path).parts[-1]
        write(join(output_dir, file_name), new_sample_rate,
              np.round(new_waveform).astype(waveform.dtype))


def preprocess(input_flac_dir: str, new_sample_rate: int, output_dir: str) -> None:
    for f in listdir(input_flac_dir):
        file_path: str = join(input_flac_dir, f)
        if isfile(file_path):
            pure_file_path: PurePath = PurePath(file_path)
            flac_tmp_audio_data = AudioSegment.from_file(pure_file_path, pure_file_path.suffix[1:])
            output_file_path = output_dir + pure_file_path.name.replace(pure_file_path.suffix, "") + ".wav"

            flac_tmp_audio_data.export(output_file_path, format="wav")
            downsample(output_file_path, new_sample_rate, output_dir)


@click.command()
@click.option("--input_dir", type=str)
@click.option("--output_dir", type=str)
@click.option("--sample_rate", type=int)
def main(input_dir: str, output_dir: str, sample_rate: int) -> None:
    preprocess(input_dir, sample_rate, output_dir)


if __name__ == "__main__":
    main()
