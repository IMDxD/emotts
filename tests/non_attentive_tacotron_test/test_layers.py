from typing import Tuple

import pytest
import torch
from torch import nn

from src.models.feature_models.layers import (
    Conv1DNorm,
    Idomp,
    LinearWithActivation,
    PositionalEncoding,
)


def test_idomp_layer() -> None:
    input_tensor = torch.randn(16, 10, 2)
    layer = Idomp()
    out = layer(input_tensor)
    assert (input_tensor == out).all(), "Layer must return tensor as it is"


@pytest.mark.parametrize(
    ("kernel_size", "output_channel", "dilation", "input_tensor", "expected_shape"),
    [
        pytest.param(3, 256, 1, torch.randn(16, 128, 24), (16, 256, 24)),
        pytest.param(5, 128, 3, torch.randn(16, 256, 32), (16, 128, 32)),
    ],
)
def test_conv_norm_layer(
    kernel_size: int,
    output_channel: int,
    dilation: int,
    input_tensor: torch.Tensor,
    expected_shape: Tuple[int, int, int],
) -> None:
    layer = Conv1DNorm(
        input_tensor.shape[1], output_channel, kernel_size, dilation=dilation
    )
    layer_out = layer(input_tensor)
    assert (
        layer_out.shape == expected_shape
    ), f"Wrong out shape, expected: {expected_shape}, got: {layer_out.shape}"


@pytest.mark.parametrize(
    ("dimension", "input_tensor", "expected_shape"),
    [
        pytest.param(32, torch.randn(16, 24, 128), (16, 24, 160)),
        pytest.param(64, torch.randn(16, 32, 256), (16, 32, 320)),
    ],
)
def test_positional_encoding_layer(
    dimension: int, input_tensor: torch.Tensor, expected_shape: Tuple[int, int, int]
) -> None:
    layer = PositionalEncoding(dimension)
    layer_out = layer(input_tensor)
    assert (
        layer_out.shape == expected_shape
    ), f"Wrong out shape, expected: {expected_shape}, got: {layer_out.shape}"


@pytest.mark.parametrize(
    ("dimension", "input_tensor", "expected_shape", "activation"),
    [
        pytest.param(32, torch.randn(16, 24, 128), (16, 24, 32), Idomp()),
        pytest.param(64, torch.randn(16, 32, 256), (16, 32, 64), nn.Softplus()),
    ],
)
def test_liner_act_layer(
    dimension: int,
    input_tensor: torch.Tensor,
    expected_shape: Tuple[int, int, int],
    activation: nn.Module,
) -> None:
    layer = LinearWithActivation(
        input_tensor.shape[-1], dimension, activation=activation
    )
    layer_out = layer(input_tensor)
    assert (
        layer_out.shape == expected_shape
    ), f"Wrong out shape, expected: {expected_shape}, got: {layer_out.shape}"


def test_liner_act_layer_relu() -> None:
    input_tensor = torch.randn(16, 24, 128)
    expected_shape = (16, 24, 32)
    layer = LinearWithActivation(128, 32, activation=nn.ReLU())
    layer_out = layer(input_tensor)
    assert (
        layer_out.shape == expected_shape
    ), f"Wrong out shape, expected: {expected_shape}, got: {layer_out.shape}"
    assert (
        layer_out[layer_out < 0].shape[0] == 0
    ), "No negative values must be after relu"
