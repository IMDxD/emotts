from __future__ import (
    absolute_import, division, print_function, unicode_literals,
)

import argparse
import json
import os

import numpy as np
import torch
from scipy.io.wavfile import write

from src.models.hifi_gan.env import AttrDict
from src.models.hifi_gan.meldataset import MAX_WAV_VALUE
from src.models.hifi_gan.models import Generator
from src.models.hifi_gan.utils import load_checkpoint


def inference(arguments: argparse.Namespace, h: AttrDict, device: torch.device) -> None:
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(arguments.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(arguments.input_mels_dir)

    os.makedirs(arguments.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            x = np.load(os.path.join(arguments.input_mels_dir, filename))
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(
                arguments.output_dir, os.path.splitext(filename)[0] + '_generated_e2e.wav'
            )
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main() -> None:
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    arguments = parser.parse_args()

    config_file = os.path.join(os.path.split(arguments.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(**json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(arguments, h, device)


if __name__ == '__main__':
    main()
