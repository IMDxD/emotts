import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AttrDict:
    resblock: str
    num_gpus: int
    batch_size: int
    learning_rate: float
    adam_b1: float
    adam_b2: float
    lr_decay: float
    seed: int
    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    upsample_initial_channel: int
    resblock_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    resblock_initial_channel: int
    segment_size: int
    num_mels: int
    num_freq: int
    n_fft: int
    hop_size: int
    win_size: int
    sampling_rate: int
    fmin: int
    fmax: int
    fmax_loss: Optional[int]
    num_workers: int
    dist_config: Dict[str, Any]


def build_env(config: str, config_name: str, path: str) -> None:
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
