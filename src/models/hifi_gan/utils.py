import glob
import os
from typing import Optional, Union

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch

from .hifi_config import HiFiGeneratorParam
from .models import Generator

matplotlib.use("Agg")


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def scan_checkpoint(cp_dir: str, prefix: str) -> Optional[str]:
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_model(
    model_path: str,
    hifi_config: HiFiGeneratorParam,
    num_mels: int,
    device: torch.device,
) -> Generator:

    cp_g = scan_checkpoint(model_path, "g_")
    generator = Generator(config=hifi_config, num_mels=num_mels).to(device)
    state_dict = torch.load(cp_g, map_location=device)
    generator.load_state_dict(state_dict["generator"])
    generator.remove_weight_norm()
    generator.eval()
    return generator
