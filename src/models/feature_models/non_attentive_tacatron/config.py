from dataclasses import dataclass, field
from typing import List


@dataclass
class TacatronDurationConfig:

    lstm_layers: int = field(default=2)
    lstm_hidden: int = field(default=256)
    dropout: float = field(default=0.5)


@dataclass
class TacatronRangeConfig:

    lstm_layers: int = field(default=2)
    lstm_hidden: int = field(default=256)
    dropout: float = field(default=0.5)


@dataclass
class GaussianUpsampleConfig:

    duration_config: TacatronDurationConfig
    range_config: TacatronRangeConfig
    eps: float = field(default=1e-20)
    positional_dim: int = field(default=32)
    teacher_forcing_ratio: float = field(default=1.0)
    attention_dropout: float = field(default=0.1)
    positional_dropout: float = field(default=0.1)


@dataclass
class TacatronDecoderConfig:

    prenet_layers: List[int] = field(default_factory=[256, 256])  # type: ignore
    decoder_rnn_dim: int = field(default=512)
    decoder_num_layers: int = field(default=3)
    teacher_forcing_ratio: float = field(default=1.0)
    p_decoder_dropout: float = field(default=0.1)


@dataclass
class TacatronEncoderConfig:

    n_convolutions: int = field(default=3)
    kernel_size: int = field(default=5)
    conv_channel: int = field(default=512)
    lstm_layers: int = field(default=1)
    lstm_hidden: int = field(default=256)


@dataclass
class TacatronPostNetConfig:

    embedding_dim: int = field(default=512)
    n_convolutions: int = field(default=5)
    kernel_size: int = field(default=5)
    dropout_rate: float = field(default=0.1)


@dataclass
class TacatronConfig:

    encoder_config: TacatronEncoderConfig
    attention_config: GaussianUpsampleConfig
    decoder_config: TacatronDecoderConfig
    postnet_config: TacatronPostNetConfig
    n_mel_channels: int = field(default=80)
    mask_padding: bool = field(default=True)
    phonem_embedding_dim: int = field(default=512)
    speaker_embedding_dim: int = field(default=256)
