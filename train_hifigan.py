import itertools
import os
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import CHECKPOINT_DIR, HIFI_CHECKPOINT_NAME, LOG_DIR, PATHLIKE
from src.models.hifi_gan.meldataset import MelDataset, mel_spectrogram
from src.models.hifi_gan.models import (
    Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
    discriminator_loss, feature_loss, generator_loss,
)
from src.models.hifi_gan.train_valid_split import split_vctk_data
from src.models.hifi_gan.utils import scan_checkpoint
from src.train_config import TrainParams

torch.backends.cudnn.benchmark = True
warnings.simplefilter(action="ignore", category=FutureWarning)


class HIFITrainer:

    def __init__(self, config: TrainParams):

        torch.cuda.manual_seed(config.seed)

        self.config = config

        self.hifi_dir = CHECKPOINT_DIR / config.checkpoint_name / HIFI_CHECKPOINT_NAME
        self.hifi_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = LOG_DIR / self.config.checkpoint_name / HIFI_CHECKPOINT_NAME
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config.device)
        self.generator = Generator(config.train_hifi.model_param, config.n_mels).to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        self.steps = 0
        self.last_epoch = -1

        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            config.train_hifi.learning_rate,
            betas=(config.train_hifi.adam_b1, config.train_hifi.adam_b2)
        )
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            config.train_hifi.learning_rate,
            betas=(config.train_hifi.adam_b1, config.train_hifi.adam_b2)
        )

        self.load_checkpoint(self.config.pretrained_hifi)

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=config.train_hifi.lr_decay, last_epoch=self.last_epoch
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=config.train_hifi.lr_decay, last_epoch=self.last_epoch
        )
        self.train_loader, self.validation_loader = self.init_data()

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.best_loss = 1e6
        self.early_stopping_rounds = 0

    def load_checkpoint(self, path: PATHLIKE):
        if os.path.isdir(path):
            cp_g = scan_checkpoint(path, "g_")
            cp_do = scan_checkpoint(path, "do_")
            state_dict_g = torch.load(cp_g, map_location=self.device)
            state_dict_do = torch.load(cp_do, map_location=self.device)
            self.generator.load_state_dict(state_dict_g["generator"])
            self.mpd.load_state_dict(state_dict_do["mpd"])
            self.msd.load_state_dict(state_dict_do["msd"])
            self.steps = state_dict_do["steps"] + 1
            self.last_epoch = state_dict_do["epoch"]
            self.optim_g.load_state_dict(state_dict_do["optim_g"])
            self.optim_d.load_state_dict(state_dict_do["optim_d"])

    def init_data(self) -> Tuple[DataLoader, DataLoader]:
        training_filelist, validation_filelist = split_vctk_data(
            self.config.data.wav_dir,
            self.config.data.feature_dir
        )

        trainset = MelDataset(
            training_files=training_filelist,
            config=self.config,
            device=self.device,
        )

        train_loader = DataLoader(
            trainset, shuffle=False, batch_size=self.config.train_hifi.batch_size,
            pin_memory=True, drop_last=True
        )

        validset = MelDataset(
            training_files=validation_filelist,
            config=self.config,
            device=self.device,
        )
        validation_loader = DataLoader(validset, shuffle=False, batch_size=1, pin_memory=True)

        return train_loader, validation_loader

    def train(self) -> None:  # noqa: C901, CCR001, CFQ001

        self.generator.train()
        self.mpd.train()
        self.msd.train()
        for epoch in range(max(0, self.last_epoch), self.config.train_hifi.training_epochs):

            for _, batch in enumerate(self.train_loader):

                x, y, _, y_mel = batch
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_mel = y_mel.to(self.device, non_blocking=True)
                y = y.unsqueeze(1)

                y_g_hat = self.generator(x)
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), self.config)

                self.optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                self.optim_d.step()

                # Generator
                self.optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                loss_gen_all.backward()
                self.optim_g.step()

                # Tensorboard summary logging
                if (self.steps + 1) % self.config.train_hifi.summary_interval == 0:
                    self.writer.add_scalar("training/gen_loss_total", loss_gen_all, self.steps)
                    self.writer.add_scalar("training/mel_spec_error", mel_error, self.steps)

                # Validation
                if self.steps % self.config.train_hifi.checkpoint_interval == 0:
                    val_loss = self.validation()
                    self.check_early_stopping(val_loss, epoch)

                self.steps += 1  # noqa: SIM113

            self.scheduler_g.step()
            self.scheduler_d.step()

    def validation(self):
        self.generator.eval()
        torch.cuda.empty_cache()
        val_err_tot = 0.0
        with torch.no_grad():
            for j, batch_val in enumerate(self.validation_loader):
                x, y, _, y_mel = batch_val
                y_g_hat = self.generator(x.to(self.device))
                y_mel = y_mel.to(self.device, non_blocking=True)
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), self.config)
                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                if j <= 4:
                    self.writer.add_audio(f"generated/y_hat_{j}", y_g_hat[0], self.steps, self.config.sample_rate)

            val_err = val_err_tot / (j + 1)
            self.writer.add_scalar("validation/mel_spec_error", val_err, self.steps)

        self.generator.train()
        return val_err

    def save(self, epoch: int) -> None:
        torch.save(
            {"generator": self.generator.state_dict()},
            self.hifi_dir / "generator.pkl"
        )

        torch.save(
            {
                "mpd": self.mpd.state_dict(),
                "msd": self.msd.state_dict(),
                "optim_g": self.optim_g.state_dict(),
                "optim_d": self.optim_d.state_dict(),
                "steps": self.steps,
                "epoch": epoch
            },
            self.hifi_dir / "discriminator.pkl"
        )

    def check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.early_stopping_rounds = 0
            self.save(epoch)
        else:
            self.early_stopping_rounds += 1
            if self.early_stopping_rounds > self.config.train_hifi.early_stopping:
                return True
        return False