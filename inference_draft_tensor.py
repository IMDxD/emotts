import json

import torch
import torchaudio

from src.models.hifi_gan.env import AttrDict
from src.models.hifi_gan.meldataset import MAX_WAV_VALUE
from src.models.hifi_gan.models import Generator
from src.models.hifi_gan.hifi_config import HIFIParams
from src.constants import MODEL_DIR
from inference_draft.wav_to_mel import mel_spectrogram
from scipy.io.wavfile import write as wav_write



def load_model(device: torch.device = torch.device("cpu")) -> Generator:
    config_path = "inference_draft/config_v1.json"
    with open(config_path) as f:
        config = AttrDict(json.load(f))
    generator = Generator(config).to(device)
    model_path = "inference_draft/generator_v1"
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict['generator'])
    generator.remove_weight_norm()
    generator.eval()
    return generator


def inference(generator: Generator, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        x = tensor.to(device)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
    return audio.type(torch.int16)


if __name__ == "__main__":
    wavetensor, sr = torchaudio.load("inference_draft/000026_RUSLAN.wav")
    mels = mel_spectrogram(wavetensor)
    audio = inference(load_model(), mels, torch.device("cpu")).detach().cpu().numpy().astype('int16')
    wav_write("inference_draft/test.wav", sr, audio)

