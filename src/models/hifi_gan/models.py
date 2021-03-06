from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import (
    remove_weight_norm as torch_remove_weight_norm, spectral_norm,
    weight_norm as torch_weight_norm,
)

from src.models.hifi_gan.hifi_config import HiFiGeneratorParam
from src.models.hifi_gan.utils import get_padding, init_weights, scan_checkpoint

LRELU_SLOPE: float = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(dilation[0],),
                        padding=(get_padding(kernel_size, dilation[0]),),
                    )
                ),
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(dilation[1],),
                        padding=(get_padding(kernel_size, dilation[1]),),
                    )
                ),
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(dilation[2],),
                        padding=(get_padding(kernel_size, dilation[2]),),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(1,),
                        padding=(get_padding(kernel_size, 1),),
                    )
                ),
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(1,),
                        padding=(get_padding(kernel_size, 1),),
                    )
                ),
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(1,),
                        padding=(get_padding(kernel_size, 1),),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            torch_remove_weight_norm(layer)
        for layer in self.convs2:
            torch_remove_weight_norm(layer)


class ResBlock2(torch.nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3)
    ) -> None:
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(dilation[0],),
                        padding=(get_padding(kernel_size, dilation[0]),),
                    )
                ),
                torch_weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        (kernel_size,),
                        (1,),
                        dilation=(dilation[1],),
                        padding=(get_padding(kernel_size, dilation[1]),),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            torch_remove_weight_norm(layer)


class Generator(torch.nn.Module):
    def __init__(self, config: HiFiGeneratorParam, num_mels: int) -> None:
        super(Generator, self).__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = torch_weight_norm(
            Conv1d(
                num_mels,
                config.upsample_initial_channel,
                (7,),
                (1,),
                padding=(3,),
            )
        )
        resblock = ResBlock1 if config.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            self.ups.append(
                torch_weight_norm(
                    ConvTranspose1d(
                        config.upsample_initial_channel // (2 ** i),
                        config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        ch = 1
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, tuple(d)))

        self.conv_post = torch_weight_norm(Conv1d(ch, 1, (7,), (1,), padding=(3,)))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs: torch.Tensor = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            torch_remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        torch_remove_weight_norm(self.conv_pre)
        torch_remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f: Any = torch_weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        y_d_rs = torch.cat(y_d_rs, dim=-1)
        y_d_gs = torch.cat(y_d_gs, dim=-1)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super(DiscriminatorS, self).__init__()
        norm_f: Any = torch_weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, (15,), (1,), padding=(7,))),
                norm_f(Conv1d(128, 128, (41,), (2,), groups=4, padding=(20,))),
                norm_f(Conv1d(128, 256, (41,), (2,), groups=16, padding=(20,))),
                norm_f(Conv1d(256, 512, (41,), (4,), groups=16, padding=(20,))),
                norm_f(Conv1d(512, 1024, (41,), (4,), groups=16, padding=(20,))),
                norm_f(Conv1d(1024, 1024, (41,), (1,), groups=16, padding=(20,))),
                norm_f(Conv1d(1024, 1024, (5,), (1,), padding=(2,))),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, (3,), (1,), padding=(1,)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        y_d_rs = torch.cat(y_d_rs, dim=-1)
        y_d_gs = torch.cat(y_d_gs, dim=-1)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(
    fmap_r: torch.Tensor, fmap_g: torch.Tensor
) -> Union[float, torch.Tensor]:
    loss: Union[float, torch.Tensor] = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: torch.Tensor, disc_generated_outputs: torch.Tensor
) -> Tuple[torch.Tensor, List[float], List[float]]:
    loss: torch.Tensor = torch.as_tensor(0.0, device=disc_real_outputs.device)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    loss: torch.Tensor = torch.as_tensor(0.0, device=disc_outputs.device)
    gen_losses = []
    for dg in disc_outputs:
        cur_loss = torch.mean((1 - dg) ** 2)
        gen_losses.append(cur_loss)
        loss += cur_loss

    return loss, gen_losses


def load_model(
    model_path: str,
    hifi_config: HiFiGeneratorParam,
    num_mels: int,
    device: torch.device,
) -> Generator:

    cp_g = scan_checkpoint(model_path, "g_")
    generator = Generator(config=hifi_config, num_mels=num_mels).to("cpu")
    state_dict = torch.load(cp_g, map_location="cpu")
    generator.load_state_dict(state_dict["generator"])
    generator.remove_weight_norm()
    generator.eval()
    generator.to(device)
    return generator
