import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .config import GSTParams
from .layers import Conv2DNorm


class GST(nn.Module):

    def __init__(self, n_mel_channels: int, config: GSTParams):

        super().__init__()
        self.encoder = ReferenceEncoder(n_mel_channels, config)
        self.stl = STL(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed


class ReferenceEncoder(nn.Module):

    def __init__(self, n_mel_channels: int, config: GSTParams):

        super().__init__()
        l_filters = len(config.ref_enc_filters)
        filters = [1] + config.ref_enc_filters
        convs = [
            Conv2DNorm(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ) for i in range(l_filters)
        ]
        self.convs = nn.Sequential(*convs)

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, l_filters)
        self.gru = nn.GRU(
            input_size=config.ref_enc_filters[-1] * out_channels,
            hidden_size=config.emb_dim // 2,
            batch_first=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        out = inputs.unsqueeze(1)
        out = self.convs(out)

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.size(0), out.size(1), -1)

        self.gru.flatten_parameters()
        _, out = self.gru(out)

        return out.squeeze(0)

    @staticmethod
    def calculate_channels(n_mels: int, kernel_size: int, stride: int, pad: int, n_convs: int):
        for i in range(n_convs):
            n_mels = (n_mels - kernel_size + 2 * pad) // stride + 1
        return n_mels


class STL(nn.Module):
    """
    inputs --- [N, E//2]
    """

    def __init__(self, config: GSTParams):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(config.token_num, config.emb_dim // config.num_heads))
        d_q = config.emb_dim // 2
        d_k = config.emb_dim // config.num_heads

        self.attention = MultiHeadAttention(
            query_dim=d_q,
            key_dim=d_k,
            num_units=config.emb_dim,
            num_heads=config.num_heads
        )

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0)
        keys = torch.repeat_interleave(keys, batch_size, dim=0)
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):

    def __init__(self, query_dim: int, key_dim: int, num_units: int, num_heads: int):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        querys = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)

        return out
