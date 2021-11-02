from torch import nn
import torch.nn.functional as F


class NonAttentiveTacotronLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        mels_weight: float,
        duration_weight: float,
    ):
        super(NonAttentiveTacotronLoss, self).__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mels_weight = mels_weight
        self.duration_weight = duration_weight

    def forward(
        self, prenet_mels, postnet_mels, model_durations, target_durations, target_mels
    ):
        target_mels.requires_grad = False
        target_durations.requires_grad = False

        prenet_l1 = F.l1_loss(prenet_mels, target_mels)
        prenet_l2 = F.mse_loss(prenet_mels, target_mels)
        postnet_l1 = F.l1_loss(postnet_mels, target_mels)
        postnet_l2 = F.mse_loss(postnet_mels, target_mels)
        loss_mels = self.mels_weight * (prenet_l1 + prenet_l2 + postnet_l1 + postnet_l2)
        loss_durations = (
            self.duration_weight
            * self.hop_size
            / self.sample_rate
            * F.mse_loss(model_durations, target_durations)
        )
        return loss_mels + loss_durations
