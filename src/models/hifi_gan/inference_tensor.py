import json
from pathlib import Path

import torch

from .hifi_config import HIFIParams, TrainParamsHiFi
from .models import Generator

MODEL_DIR = Path("models")


def inference(generator: Generator, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:

    with torch.no_grad():
        x = tensor.unsqueeze(0).to(device)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
    return audio
