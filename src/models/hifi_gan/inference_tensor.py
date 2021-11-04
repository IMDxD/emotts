import json

import torch

from .env import AttrDict
from .meldataset import MAX_WAV_VALUE
from .models import Generator
from .hifi_config import HIFIParams
from src.constants import MODEL_DIR


def load_model(config: HIFIParams, device: torch.device) -> Generator:
    dir_path = MODEL_DIR / config.dir_path
    config_path = dir_path / config.config_name
    model_path = dir_path / config.model_name
    with open(config_path) as f:
        config = AttrDict(json.load(f))
    generator = Generator(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict['generator'])
    generator.remove_weight_norm()
    generator.eval()
    return generator


def inference(generator: Generator, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:

    with torch.no_grad():
        x = tensor.unsqueeze(0).to(device)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
    return audio.type(torch.int16)
