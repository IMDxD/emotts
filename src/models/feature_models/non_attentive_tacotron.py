import random
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as f

from src.data_process import VCTKBatch
from .config import (
    DecoderParams, DurationParams, EncoderParams, GaussianUpsampleParams,
    ModelParams, PostNetParams, RangeParams,
)
from .layers import (
    Conv1DNorm, LinearWithActivation, PositionalEncoding,
)
from .utils import (
    get_mask_from_lengths, norm_emb_layer,
)
from .gst import GST


class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: List[int], dropout: float):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearWithActivation(
                    in_size, out_size, bias=False, activation="SoftPlus"
                )
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.layers:
            x = f.dropout(f.relu(linear(x)), p=self.dropout, training=self.training)
        return x


class Postnet(nn.Module):
    def __init__(self, n_mel_channels: int, config: PostNetParams):
        super().__init__()
        self.dropout = config.dropout
        convolutions: List[nn.Module] = [
            Conv1DNorm(
                n_mel_channels,
                config.embedding_dim,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                dropout_rate=config.dropout,
                w_init_gain="tanh",
            ),
            nn.Tanh()
        ]

        for _ in range(config.n_convolutions - 2):
            convolutions.append(
                Conv1DNorm(
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
            Conv1DNorm(
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
    def __init__(self, embedding_dim: int, config: DurationParams):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = LinearWithActivation(config.lstm_hidden * 2, 1, bias=False)

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
    def __init__(self, embedding_dim: int, config: RangeParams):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim + 1,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = LinearWithActivation(config.lstm_hidden * 2, 1)

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
        self, embedding_dim: int, config: GaussianUpsampleParams
    ):
        super().__init__()
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.eps = config.eps
        self.dropout = config.attention_dropout

        self.duration_predictor = DurationPredictor(
            embedding_dim, config.duration_config
        )
        self.range_predictor = RangePredictor(embedding_dim, config.range_config)
        self.positional_encoder = PositionalEncoding(
            config.positional_dim,
            dropout=config.positional_dropout
        )

    def calc_scores(
        self, durations: torch.Tensor, ranges: torch.Tensor
    ) -> torch.Tensor:
        # Calc gaussian weight for Gaussian upsampling attention
        duration_cumsum = durations.cumsum(dim=1).float()
        max_duration = duration_cumsum[:, -1, :].max().long()
        c = duration_cumsum - 0.5 * durations
        t = torch.arange(0, max_duration.item()).view(1, 1, -1).to(ranges.device)

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

        embeddings_per_duration = torch.matmul(scores.transpose(1, 2), embeddings)
        embeddings_per_duration = self.positional_encoder(embeddings_per_duration)
        return durations.squeeze(2), embeddings_per_duration

    def inference(
        self, embeddings: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:

        durations = self.duration_predictor(embeddings, input_lengths)
        ranges = self.range_predictor(embeddings, durations, input_lengths)

        scores = self.calc_scores(durations, ranges)

        embeddings_per_duration = torch.matmul(scores.transpose(1, 2), embeddings)
        embeddings_per_duration = self.positional_encoder(embeddings_per_duration)
        return embeddings_per_duration


class Encoder(nn.Module):
    def __init__(self, phonem_embedding_dim: int, config: EncoderParams):
        super().__init__()

        convolutions: List[nn.Module] = [
            Conv1DNorm(
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
            conv_layer = Conv1DNorm(
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
            Conv1DNorm(
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
        self, n_mel_channels: int,
            n_frames_per_step: int,
            attention_out_dim: int,
            config: DecoderParams
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.decoder_rnn_dim = config.decoder_rnn_dim
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.p_decoder_dropout = config.dropout
        self.n_frames_per_step = n_frames_per_step

        self.prenet = Prenet(
            self.n_mel_channels,
            config.prenet_layers,
            config.prenet_dropout,
        )

        self.decoder_rnn = nn.LSTM(
            (attention_out_dim + config.prenet_layers[-1]) * self.n_frames_per_step,
            config.decoder_rnn_dim * self.n_frames_per_step,
            num_layers=config.decoder_num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.linear_projection = LinearWithActivation(
            (config.decoder_rnn_dim + attention_out_dim) * self.n_frames_per_step,
            n_mel_channels * self.n_frames_per_step
        )

    def forward(self, memory: torch.Tensor, y_mels: torch.Tensor) -> torch.Tensor:

        batch_size = memory.shape[0]
        previous_frame = torch.zeros(
            memory.shape[0],
            1,
            self.n_mel_channels * self.n_frames_per_step
        ).to(memory.device)
        to_pad_mels = self.n_frames_per_step - y_mels.shape[1] % self.n_frames_per_step
        padded_y_mels = f.pad(
            y_mels.permute(0, 2, 1),
            (0, to_pad_mels),
            mode="constant",
            value=0
        ).permute(0, 2, 1)
        padded_memory = f.pad(
            memory.permute(0, 2, 1),
            (0, to_pad_mels),
            mode="constant",
            value=0
        ).permute(0, 2, 1)
        new_len = padded_memory.shape[1] // self.n_frames_per_step
        padded_y_mels = padded_y_mels.reshape(batch_size, new_len, -1)
        padded_y_mels = torch.cat((previous_frame, padded_y_mels[:, :-1, :]), dim=1)
        padded_memory = padded_memory.reshape(batch_size, new_len, -1)
        previous_frame = previous_frame[:, 0, :]

        mel_outputs = []
        decoder_state = None

        for i in range(new_len):
            previous_frame = self.prenet(
                previous_frame.view(
                    batch_size,
                    -1,
                    self.n_mel_channels
                )
            )
            decoder_input: torch.Tensor = torch.cat(
                (previous_frame.view(batch_size, -1), padded_memory[:, i, :]), dim=-1
            )
            out, decoder_state = self.decoder_rnn(
                decoder_input.unsqueeze(1), decoder_state
            )
            out = torch.cat((out, padded_memory[:, i, :].unsqueeze(1)), dim=-1)
            mel_out = self.linear_projection(out)
            mel_outputs.append(mel_out)
            if random.uniform(0, 1) > self.teacher_forcing_ratio:
                previous_frame = mel_out.squeeze(1)
            else:
                previous_frame = padded_y_mels[:, i, :]

        mel_tensor_outputs: torch.Tensor = torch.cat(mel_outputs, dim=1)
        mel_tensor_outputs = mel_tensor_outputs.reshape(batch_size, -1, self.n_mel_channels)
        return mel_tensor_outputs[:, :-to_pad_mels, :]

    def inference(self, memory: torch.Tensor) -> torch.Tensor:

        batch_size = memory.shape[0]
        to_pad_mels = self.n_frames_per_step - memory.shape[1] % self.n_frames_per_step
        previous_frame = torch.zeros(
            memory.shape[0],
            self.n_mel_channels * self.n_frames_per_step
        ).to(memory.device)
        padded_memory = f.pad(
            memory.permute(0, 2, 1),
            (0, to_pad_mels),
            mode="constant",
            value=0
        ).permute(0, 2, 1)
        new_len = padded_memory.shape[1] // self.n_frames_per_step
        padded_memory = padded_memory.reshape(batch_size, new_len, -1)

        mel_outputs = []
        decoder_state = None

        for i in range(new_len):
            previous_frame = self.prenet(
                previous_frame.view(
                    batch_size,
                    -1,
                    self.n_mel_channels
                )
            )
            decoder_input: torch.Tensor = torch.cat(
                (previous_frame.view(batch_size, -1), padded_memory[:, i, :]), dim=-1
            )
            out, decoder_state = self.decoder_rnn(
                decoder_input.unsqueeze(1), decoder_state
            )
            out = torch.cat((out, padded_memory[:, i, :].unsqueeze(1)), dim=-1)
            mel_out = self.linear_projection(out)
            mel_outputs.append(mel_out)
            previous_frame = mel_out.squeeze(1)

        mel_tensor_outputs: torch.Tensor = torch.cat(mel_outputs, dim=1)
        mel_tensor_outputs = mel_tensor_outputs.reshape(batch_size, -1, self.n_mel_channels)
        return mel_tensor_outputs[:, :memory.shape[1], :]


class NonAttentiveTacotron(nn.Module):
    def __init__(
        self,
        n_phonems: int,
        n_speakers: int,
        n_mel_channels: int,
        config: ModelParams,
    ):
        super().__init__()

        full_embedding_dim = config.phonem_embedding_dim + config.speaker_embedding_dim
        self.phonem_embedding = nn.Embedding(n_phonems, config.phonem_embedding_dim, padding_idx=0)
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
            config.attention_config
        )
        self.gst = GST(
            n_mel_channels=n_mel_channels,
            config=config.gst_config
        )
        styled_attention_dim = (
            full_embedding_dim +
            config.attention_config.positional_dim +
            config.gst_config.emb_dim
        )
        self.decoder = Decoder(
            n_mel_channels,
            config.n_frames_per_step,
            styled_attention_dim,
            config.decoder_config
        )
        self.postnet = Postnet(
            n_mel_channels,
            config.postnet_config,
        )

    def forward(
        self, batch: VCTKBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        phonem_emb = self.phonem_embedding(batch.phonemes).transpose(1, 2)
        speaker_emb = self.speaker_embedding(batch.speaker_ids).unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, batch.num_phonemes)

        speaker_emb = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, speaker_emb), dim=-1)

        durations, attented_embeddings = self.attention(
            embeddings, batch.num_phonemes, batch.durations
        )
        style_emb = self.gst(batch.mels)
        style_emb = torch.repeat_interleave(style_emb, attented_embeddings.shape[1], dim=1)
        styled_attention_embedding = torch.cat([attented_embeddings, style_emb], dim=-1)
        mel_outputs = self.decoder(styled_attention_embedding, batch.mels)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)
        mask = get_mask_from_lengths(
            batch.durations.cumsum(dim=1)[:, -1].long(), device=batch.phonemes.device
        )
        mel_outputs_postnet[mask] = 0
        mel_outputs[mask] = 0

        return durations, mel_outputs_postnet, mel_outputs

    def inference(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        text_inputs, text_lengths, speaker_ids, reference_mel = batch
        phonem_emb = self.phonem_embedding(text_inputs).transpose(1, 2)
        speaker_emb = self.speaker_embedding(speaker_ids).unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, text_lengths)

        speaker_emb = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, speaker_emb), dim=-1)

        attented_embeddings = self.attention.inference(embeddings, text_lengths)
        style_emb = self.gst(reference_mel)
        style_emb = torch.repeat_interleave(style_emb, attented_embeddings.shape[1], dim=1)
        styled_attention_embedding = torch.cat([attented_embeddings, style_emb], dim=-1)
        mel_outputs = self.decoder.inference(styled_attention_embedding)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)

        return mel_outputs_postnet
