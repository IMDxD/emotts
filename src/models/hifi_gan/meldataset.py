import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read
from torchaudio.transforms import Resample

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
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: Optional[int],
    center: bool = False,
) -> torch.Tensor:
    if torch.min(y) < -1.0:
        print(f"min value is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"max value is {torch.max(y)}")

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
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
        base_mels_path: str,
        segment_size: int,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        sampling_rate: int,
        fmin: int,
        fmax: int,
        fmax_loss: Optional[int],
        split: bool = True,
        shuffle: bool = True,
        n_cache_reuse: int = 1,
        device: Optional[torch.device] = None,
        fine_tuning: bool = False,
        random_seed: int = 1234,
    ) -> None:
        self.audio_files = training_files
        random.seed(random_seed)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav: Optional[AudioData]
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        filename = self.audio_files[index]
        raw_audio: Optional[AudioData]
        if self._cache_ref_count == 0:
            raw_audio, sampling_rate = torchaudio.load(filename)
            raw_audio = raw_audio.squeeze(0)

            if not self.fine_tuning:
                raw_audio = normalize(raw_audio) * 0.95

            self.cached_wav = raw_audio
            if sampling_rate != self.sampling_rate:
                resampler = Resample(orig_freq=sampling_rate,
                                     new_freq=self.sampling_rate,
                                     resampling_method='sinc_interpolation'
                                     )
                raw_audio = resampler(raw_audio)
            self._cache_ref_count = self.n_cache_reuse
        else:
            raw_audio = self.cached_wav
            self._cache_ref_count -= 1

        audio: torch.Tensor = torch.FloatTensor(raw_audio).unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start: audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

            mel = mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )
        else:

            filename = get_mel_file_path(filename, self.base_mels_path)
            mel = torch.load(filename, map_location="cpu")

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    frame_max = max(0,
                                    min(mel.size(2) - frames_per_seg - 1, 
                                        int((audio.size(1) - self.segment_size) / self.hop_size) - 1)
                                    )
                    mel_start = random.randint(0, frame_max)
                    mel = mel[:, :, mel_start: mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size,
                    ]
                else:
                    mel = torch.nn.functional.pad(
                        mel, (0, frames_per_seg - mel.size(2)), "constant"
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

        mel_loss = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        if mel_loss.size(2) > mel.size(2):
            mel = torch.nn.functional.pad(
                        mel, (0, mel_loss.size(2) - mel.size(2)), 'constant'
                    )
        elif mel_loss.size(2) < mel.size(2):
            mel_loss = torch.nn.functional.pad(
                        mel_loss, (0, mel.size(2) - mel_loss.size(2)), 'constant'
                    )

        return mel.squeeze().detach(), audio.squeeze(0).detach(), str(filename), mel_loss.squeeze().detach()

    def __len__(self) -> int:
        return len(self.audio_files)
