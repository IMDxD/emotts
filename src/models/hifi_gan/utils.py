import glob
import os
from typing import Any, Dict, Union

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch

matplotlib.use("Agg")


def plot_spectrogram(
        spectrogram: Union[np.ndarray[int, np.dtype[np.float32]], torch.Tensor]
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)  # type: ignore


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath: str, device: torch.device) -> Dict[str, Any]:
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict: Dict[str, Any] = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath: str, obj: Dict[str, Union[int, Dict[str, torch.Tensor]]]) -> None:
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir: str, prefix: str) -> str:
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]
