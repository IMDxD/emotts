import argparse
import math
import os
import random
from typing import Any, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read
from torchaudio.transforms import Resample

from src.train_config import TrainParams
from src.models.hifi_gan.train_valid_split import get_mel_file_path

AudioData = np.ndarray


def load_wav(full_path: Union[str, TextIO]) -> Tuple[AudioData, int]:
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(
    x: Any,
    c: int = 1,
    clip_val: float = 1e-5,
) -> np.ndarray:
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * c)


def dynamic_range_decompression(x: np.ndarray, c: int = 1) -> np.ndarray:
    return np.exp(x) / c


def dynamic_range_compression_torch(
    x: torch.Tensor, c: int = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * c)


def dynamic_range_decompression_torch(x: torch.Tensor, c: int = 1) -> torch.Tensor:
    return torch.exp(x) / c


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y: torch.Tensor,
    config: TrainParams,
    center: bool = False,
) -> torch.Tensor:
    if torch.min(y) < -1.0:
        print(f"min value is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"max value is {torch.max(y)}")

    global mel_basis, hann_window
    if config.f_max not in mel_basis:
        mel = librosa_mel_fn(
            config.sample_rate, config.n_fft, config.n_mels, config.f_min, config.f_max
        )
        mel_basis[str(config.f_max) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(config.win_size).to(y.device)

    y = F.pad(
        y.unsqueeze(1),
        (
            int((config.n_fft - config.hop_size) / 2),
            int((config.n_fft - config.hop_size) / 2),
        ),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        config.n_fft,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[str(config.f_max) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(arguments: argparse.Namespace) -> Tuple[List[str], List[str]]:
    with open(arguments.input_training_file, "r", encoding="utf-8") as fi:
        training_files = [
            os.path.join(arguments.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]

    with open(arguments.input_validation_file, "r", encoding="utf-8") as fi:
        validation_files = [
            os.path.join(arguments.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
    return training_files, validation_files


class MelDataset(
    torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]]
):
    def __init__(  # noqa: CFQ002
        self,
        training_files: List[str],
        config: TrainParams,
        device: Optional[torch.device] = None,
    ) -> None:

        self.audio_files = training_files
        self.config = config
        self.base_mels_path = config.data.feature_dir
        self.segment_size = config.train_hifi.segment_size
        self.fine_tuning = config.train_hifi.fine_tuning
        self.split = config.train_hifi.split_data
        self.device = device

    def __getitem__(  # noqa: CCR001
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        filename = self.audio_files[index]
        raw_audio: Optional[AudioData]

        raw_audio, sampling_rate = torchaudio.load(filename)
        raw_audio = raw_audio.squeeze(0)

        if not self.fine_tuning:
            raw_audio = normalize(raw_audio) * 0.95

        if sampling_rate != self.config.sample_rate:
            resampler = Resample(
                orig_freq=sampling_rate,
                new_freq=self.config.sample_rate,
                resampling_method='sinc_interpolation',
            )
            raw_audio = resampler(raw_audio)

        audio = torch.FloatTensor(raw_audio).unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start: audio_start + self.segment_size]
                else:
                    audio = F.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

            mel = mel_spectrogram(
                audio,
                self.config,
                center=False,
            )
        else:

            filename = get_mel_file_path(filename, self.base_mels_path)
            mel = torch.load(filename, map_location="cpu")

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.config.hop_size)

                if audio.size(1) >= self.segment_size:
                    frame_max = max(
                        0,
                        min(
                            mel.shape[2] - frames_per_seg - 1,
                            int(
                                (audio.size(1) - self.segment_size)
                                / self.config.hop_size
                            )
                            - 1,
                        ),
                    )
                    mel_start = random.randint(0, frame_max)
                    mel = mel[:, :, mel_start: mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start
                        * self.config.hop_size: (mel_start + frames_per_seg)
                        * self.config.hop_size,
                    ]
                else:
                    mel = F.pad(mel, (0, frames_per_seg - mel.shape[2]), "constant")
                    audio = F.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

        mel_loss = mel_spectrogram(
            audio,
            self.config,
            center=False,
        )

        if mel_loss.shape[2] > mel.shape[2]:
            mel = F.pad(mel, (0, mel_loss.shape[2] - mel.shape[2]), 'constant')
        elif mel_loss.shape[2] < mel.shape[2]:
            mel_loss = F.pad(
                mel_loss, (0, mel.shape[2] - mel_loss.shape[2]), 'constant'
            )

        return (
            mel.squeeze().detach(),
            audio.squeeze(0).detach(),
            str(filename),
            mel_loss.squeeze().detach(),
        )

    def __len__(self) -> int:
        return len(self.audio_files)
