from math import sqrt
import numpy as np
import torch
from scipy.io.wavfile import read


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids >= lengths.unsqueeze(1))
    return mask


def norm_emb(emb, n_symbols, embedding_dim):
    std = sqrt(2.0 / (n_symbols + embedding_dim))
    val = sqrt(3.0) * std  # uniform bounds for std
    emb.weight.data.uniform_(-val, val)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
