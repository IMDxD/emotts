import json

import torch

from src.constants import MODEL_DIR
from src.models.hifi_gan.env import AttrDict
from src.models.hifi_gan.hifi_config import HIFIParams
from src.models.hifi_gan.meldataset import MAX_WAV_VALUE
from src.models.hifi_gan.models import Generator


def load_model(hifi_config: HIFIParams, device: torch.device) -> Generator:
    dir_path = MODEL_DIR / hifi_config.dir_path
    config_path = dir_path / hifi_config.config_name
    model_path = dir_path / hifi_config.model_name
    with open(config_path) as f:
        config = AttrDict(**json.load(f))
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
    return audio
