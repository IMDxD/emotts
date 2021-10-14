import torch

from src.models.feature_models.non_attentive_tacatron.config import (
    DecoderConfig, DurationConfig, EncoderConfig, GaussianUpsampleConfig,
    ModelConfig, PostNetConfig, RangeConfig,
)
from src.models.feature_models.non_attentive_tacatron.model import (
    Attention, DurationPredictor, Encoder, RangePredictor,
)

DECODER_CONFIG = DecoderConfig()
DURATION_CONFIG = DurationConfig()
RANGE_CONFIG = RangeConfig()
ENCODER_CONFIG = EncoderConfig()
ATTENTION_CONFIG = GaussianUpsampleConfig(
    duration_config=DURATION_CONFIG, range_config=RANGE_CONFIG
)
POSTNET_CONFIG = PostNetConfig()
MODEL_CONFIG = ModelConfig(
    encoder_config=ENCODER_CONFIG,
    attention_config=ATTENTION_CONFIG,
    decoder_config=DECODER_CONFIG,
    postnet_config=POSTNET_CONFIG,
)
EMBEDDING_DIM = MODEL_CONFIG.phonem_embedding_dim + MODEL_CONFIG.speaker_embedding_dim
INPUT_PHONEMS = torch.randint(100, size=(16, 50), dtype=torch.long)
INPUT_SPEAKERS = torch.randint(4, size=(16,), dtype=torch.long)
PHONEM_EMB = torch.randn(16, 50, MODEL_CONFIG.phonem_embedding_dim, dtype=torch.float)
SPEAKER_EMB = torch.randn(16, MODEL_CONFIG.speaker_embedding_dim, dtype=torch.float)
EMBEDDING = torch.cat(
    [PHONEM_EMB, torch.repeat_interleave(SPEAKER_EMB.unsqueeze(1), 50, dim=1)], dim=-1
)
INPUT_LENGTH = torch.arange(35, 51, dtype=torch.long)
INPUT_DURATIONS = torch.randint(5, 10, size=(16, 50), dtype=torch.long)
for i, l in enumerate(INPUT_LENGTH):
    INPUT_DURATIONS[i, l:] = 0
DURATIONS_MAX = INPUT_DURATIONS.cumsum(dim=1).max(dim=1).values
INPUT_MELS = torch.randn(
    16, DURATIONS_MAX.max(), MODEL_CONFIG.n_mel_channels, dtype=torch.float
)
for i, l in enumerate(DURATIONS_MAX):
    INPUT_MELS[i, l:, :] = 0


def test_encoder_layer():
    expected_shape = (16, 50, MODEL_CONFIG.phonem_embedding_dim)
    layer = Encoder(ModelConfig.phonem_embedding_dim, config=ENCODER_CONFIG)
    out = layer(PHONEM_EMB.transpose(1, 2), INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (out[idx, length - 1] != 0).any(), f"Wrong zero vector for id = {idx}"


def test_duration_layer():
    expected_shape = (16, 50, 1)
    layer = DurationPredictor(EMBEDDING_DIM, config=DURATION_CONFIG)
    zero_value = layer.projection.linear_layer.bias
    if zero_value is None:
        zero_value = 0
    out = layer(EMBEDDING, INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == zero_value
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            out[idx, length - 1] != zero_value
        ).any(), f"Wrong zero vector for id = {idx}"


def test_range_layer():
    expected_shape = (16, 50, 1)
    layer = RangePredictor(EMBEDDING_DIM, config=RANGE_CONFIG)
    zero_value = layer.projection.linear_layer.bias
    if zero_value is None:
        zero_value = 0
    out = layer(EMBEDDING, INPUT_DURATIONS.unsqueeze(2), INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == zero_value
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            out[idx, length - 1] != zero_value
        ).any(), f"Wrong zero vector for id = {idx}"


def test_attention_layer():
    expected_shape_out = (
        16,
        DURATIONS_MAX.max().item(),
        EMBEDDING_DIM + ATTENTION_CONFIG.positional_dim,
    )
    expected_shape_dur = (16, 50, 1)
    layer = Attention(
        EMBEDDING_DIM, config=ATTENTION_CONFIG, device=torch.device("cpu")
    )
    dur, out = layer(EMBEDDING, INPUT_LENGTH, INPUT_DURATIONS)
    assert (
        dur.shape == expected_shape_dur
    ), f"Wrong shape, expected {expected_shape_dur}, got: {dur.shape}"
    assert (
        out.shape == expected_shape_out
    ), f"Wrong shape, expected {expected_shape_out}, got: {out.shape}"
    for idx, length in enumerate(DURATIONS_MAX):
        assert (
            out[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (out[idx, length - 1] != 0).any(), f"Wrong zero vector for id = {idx}"
