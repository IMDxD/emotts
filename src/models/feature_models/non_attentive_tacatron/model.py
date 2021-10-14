import random
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as f

from .config import (
    DecoderConfig, DurationConfig, EncoderConfig, GaussianUpsampleConfig,
    ModelConfig, PostNetConfig, RangeConfig,
)
from .layers import ConvNorm, LinearNorm, PositionalEncoding
from .utils import get_mask_from_lengths, norm_emb_layer


class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: List[int], dropout: float):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.layers:
            x = f.dropout(f.relu(linear(x)), p=self.dropout, training=self.training)
        return x


class Postnet(nn.Module):
    def __init__(self, n_mel_channels: int, config: PostNetConfig):
        super().__init__()
        self.dropout = config.dropout
        convolutions: List[nn.Module] = []

        convolutions.append(
            ConvNorm(
                n_mel_channels,
                config.embedding_dim,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                dropout_rate=config.dropout,
                w_init_gain="tanh",
            )
        )
        convolutions.append(nn.Tanh())

        for _ in range(config.n_convolutions - 2):
            convolutions.append(
                ConvNorm(
                    config.embedding_dim,
                    config.embedding_dim,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=int((config.kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                )
            )
            convolutions.append(nn.Tanh())

        convolutions.append(
            ConvNorm(
                config.embedding_dim,
                n_mel_channels,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="linear",
            )
        )
        self.convolutions = nn.Sequential(*convolutions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)

        return x


class DurationPredictor(nn.Module):
    def __init__(self, embedding_dim: int, config: DurationConfig):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = LinearNorm(config.lstm_hidden * 2, 1, bias=False)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = f.dropout(outputs, self.dropout, self.training)
        x = self.projection(outputs)
        return x


class RangePredictor(nn.Module):
    def __init__(self, embedding_dim: int, config: RangeConfig):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim + 1,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = LinearNorm(config.lstm_hidden * 2, 1)

    def forward(
        self, x: torch.Tensor, durations: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:

        x = torch.cat((x, durations), dim=-1)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = f.dropout(outputs, self.dropout, self.training)
        outputs = self.projection(outputs)
        return outputs


class Attention(nn.Module):
    def __init__(
        self, embedding_dim: int, config: GaussianUpsampleConfig, device: torch.device
    ):
        super().__init__()
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.device = device
        self.eps = config.eps
        self.dropout = config.attention_dropout

        self.duration_predictor = DurationPredictor(
            embedding_dim, config.duration_config
        )
        self.range_predictor = RangePredictor(embedding_dim, config.range_config)
        self.positional_encoder = PositionalEncoding(
            config.positional_dim,
            dropout=config.positional_dropout,
        )

    def calc_scores(
        self, durations: torch.Tensor, ranges: torch.Tensor
    ) -> torch.Tensor:

        duration_cumsum = durations.cumsum(dim=1).float()
        max_duration = duration_cumsum[:, -1, :].max()
        c = duration_cumsum - 0.5 * durations
        t = torch.arange(0, max_duration.item()).view(1, 1, -1).to(self.device)

        weights = torch.exp(-(ranges ** -2) * ((t - c) ** 2))
        weights_norm = torch.sum(weights, dim=1, keepdim=True) + self.eps
        weights = weights / weights_norm

        return weights

    def forward(
        self,
        embeddings: torch.Tensor,
        input_lengths: torch.Tensor,
        y_durations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_lengths = input_lengths.cpu().numpy()

        durations = self.duration_predictor(embeddings, input_lengths)
        ranges = self.range_predictor(embeddings, durations, input_lengths)

        if random.uniform(0, 1) > self.teacher_forcing_ratio:  # type: ignore
            scores = self.calc_scores(durations, ranges)
        else:
            scores = self.calc_scores(y_durations.unsqueeze(2), ranges)

        attented_embeddings = torch.matmul(scores.transpose(1, 2), embeddings)
        attented_embeddings = self.positional_encoder(attented_embeddings)
        mask = get_mask_from_lengths(y_durations.cumsum(dim=1)[:, -1], device=self.device)
        attented_embeddings[mask] = 0
        return durations, attented_embeddings


class Encoder(nn.Module):
    def __init__(self, phonem_embedding_dim: int, config: EncoderConfig):
        super().__init__()

        convolutions: List[nn.Module] = [
            ConvNorm(
                phonem_embedding_dim,
                config.conv_channel,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                dropout_rate=config.dropout,
                w_init_gain="relu",
            )
        ]

        for _ in range(config.n_convolutions - 2):
            conv_layer = ConvNorm(
                config.conv_channel,
                config.conv_channel,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            )
            convolutions.append(conv_layer)

        convolutions.append(
            ConvNorm(
                config.conv_channel,
                phonem_embedding_dim,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
        )
        self.convolutions = nn.Sequential(*convolutions)
        self.lstm = nn.LSTM(
            phonem_embedding_dim,
            config.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self, phonem_emb: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:

        phonem_emb = self.convolutions(phonem_emb)
        phonem_emb = phonem_emb.transpose(1, 2)
        phonem_emb_packed = nn.utils.rnn.pack_padded_sequence(
            phonem_emb, input_lengths, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        phonem_emb_packed, _ = self.lstm(phonem_emb_packed)

        phonem_emb, _ = nn.utils.rnn.pad_packed_sequence(
            phonem_emb_packed, batch_first=True
        )

        return phonem_emb


class Decoder(nn.Module):
    def __init__(
        self, n_mel_channels: int, attention_out_dim: int, config: DecoderConfig
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.decoder_rnn_dim = config.decoder_rnn_dim
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.p_decoder_dropout = config.dropout

        self.prenet = Prenet(
            n_mel_channels,
            config.prenet_layers,
            config.prenet_dropout,
        )

        self.decoder_rnn = nn.LSTM(
            attention_out_dim + config.prenet_layers[-1],
            config.decoder_rnn_dim,
            num_layers=config.decoder_num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.linear_projection = LinearNorm(
            config.decoder_rnn_dim + attention_out_dim, n_mel_channels
        )

    def forward(self, memory: torch.Tensor, y_mels: torch.Tensor) -> torch.Tensor:

        previous_frame = torch.zeros(memory.shape[0], 1, self.n_mel_channels)
        y_mels = torch.cat((previous_frame, y_mels[:, :-1, :]), dim=1)
        previous_frame = previous_frame[:, 0, :]

        mel_outputs = []
        decoder_state = None

        for i in range(memory.shape[1]):
            previous_frame = self.prenet(previous_frame)
            decoder_input: torch.Tensor = torch.cat(
                (previous_frame, memory[:, i, :]), dim=-1
            )
            out, decoder_state = self.decoder_rnn(
                decoder_input.unsqueeze(1), decoder_state
            )
            out = torch.cat((out, memory[:, i, :].unsqueeze(1)), dim=-1)
            mel_out = self.linear_projection(out)
            mel_outputs.append(mel_out)
            if random.uniform(0, 1) > self.teacher_forcing_ratio:
                previous_frame = mel_out.squeeze(1)
            else:
                previous_frame = y_mels[:, i, :]

        mel_tensor_outputs: torch.Tensor = torch.cat(mel_outputs, dim=1)
        return mel_tensor_outputs


class NonAttentiveTacatron(nn.Module):
    def __init__(
        self,
        n_phonems: int,
        n_speakers: int,
        device: torch.device,
        config: ModelConfig,
    ):
        super().__init__()
        full_embedding_dim = config.phonem_embedding_dim + config.speaker_embedding_dim
        self.phonem_embedding = nn.Embedding(n_phonems, config.phonem_embedding_dim)
        self.speaker_embedding = nn.Embedding(
            n_speakers,
            config.speaker_embedding_dim,
        )
        norm_emb_layer(
            self.phonem_embedding,
            n_phonems,
            config.phonem_embedding_dim,
        )
        norm_emb_layer(
            self.speaker_embedding,
            n_speakers,
            config.speaker_embedding_dim,
        )
        self.encoder = Encoder(
            config.phonem_embedding_dim,
            config.encoder_config,
        )
        self.attention = Attention(
            full_embedding_dim,
            config.attention_config,
            torch.device(device),
        )
        self.decoder = Decoder(
            config.n_mel_channels,
            full_embedding_dim + config.attention_config.positional_dim,
            config.decoder_config,
        )
        self.postnet = Postnet(
            config.n_mel_channels,
            config.postnet_config,
        )

    def forward(
        self,
        inputs: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_inputs, text_lengths, speaker_ids, y_durations, y_mels = inputs

        phonem_emb = self.phonem_embedding(text_inputs).transpose(1, 2)
        speaker_emb = self.speaker_embedding(speaker_ids).unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, text_lengths)

        speaker_emb = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, speaker_emb), dim=-1)

        durations, attented_embeddings = self.attention(
            embeddings, text_lengths, y_durations
        )
        mel_outputs = self.decoder(attented_embeddings, y_mels)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)

        return durations, mel_outputs_postnet, mel_outputs
