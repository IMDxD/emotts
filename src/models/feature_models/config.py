from dataclasses import dataclass, field
from typing import List


@dataclass
class DurationParams:

    lstm_layers: int = field(default=2)
    lstm_hidden: int = field(default=256)
    dropout: float = field(default=0.5)


@dataclass
class RangeParams:

    lstm_layers: int = field(default=2)
    lstm_hidden: int = field(default=256)
    dropout: float = field(default=0.5)


@dataclass
class GSTParams:

    ref_enc_filters: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 128, 128])
    emb_dim: int = field(default=256)
    num_heads: int = field(default=8)
    token_num: int = field(default=10)


@dataclass
class GaussianUpsampleParams:

    duration_config: DurationParams
    range_config: RangeParams
    eps: float = field(default=1e-6)
    positional_dim: int = field(default=32)
    teacher_forcing_ratio: float = field(default=1.0)
    attention_dropout: float = field(default=0.1)
    positional_dropout: float = field(default=0.0)


@dataclass
class DecoderParams:

    prenet_layers: List[int] = field(default_factory=lambda: [256, 256])
    prenet_dropout: float = field(default=0.5)
    decoder_rnn_dim: int = field(default=512)
    decoder_num_layers: int = field(default=3)
    teacher_forcing_ratio: float = field(default=1.0)
    dropout: float = field(default=0.1)


@dataclass
class EncoderParams:

    n_convolutions: int = field(default=3)
    kernel_size: int = field(default=5)
    conv_channel: int = field(default=512)
    lstm_layers: int = field(default=1)
    lstm_hidden: int = field(default=256)
    dropout: float = field(default=0.1)


@dataclass
class PostNetParams:

    embedding_dim: int = field(default=512)
    n_convolutions: int = field(default=5)
    kernel_size: int = field(default=5)
    dropout: float = field(default=0.1)


@dataclass
class ModelParams:

    encoder_config: EncoderParams
    attention_config: GaussianUpsampleParams
    decoder_config: DecoderParams
    postnet_config: PostNetParams
    gst_config: GSTParams
    n_frames_per_step: int = field(default=3)
    mask_padding: bool = field(default=True)
    phonem_embedding_dim: int = field(default=512)
    speaker_embedding_dim: int = field(default=256)
