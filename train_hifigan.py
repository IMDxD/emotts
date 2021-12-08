import argparse
import itertools
import json
import os
import time
import warnings
from typing import Optional, Union

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.models.hifi_gan.env import AttrDict, build_env
from src.models.hifi_gan.meldataset import (
    MelDataset, mel_spectrogram,
)
from src.models.hifi_gan.models import (
    Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
    discriminator_loss, feature_loss, generator_loss,
)
from src.models.hifi_gan.utils import (
    load_checkpoint, plot_spectrogram, save_checkpoint, scan_checkpoint,
)
from src.models.hifi_gan.train_valid_split import split_vctk_data

torch.backends.cudnn.benchmark = True
warnings.simplefilter(action="ignore", category=FutureWarning)


def train(rank: int, arguments: argparse.Namespace, h: AttrDict) -> None:  # noqa: C901, CCR001, CFQ001
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config["dist_backend"], init_method=h.dist_config["dist_url"],
                           world_size=h.dist_config["world_size"] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device(f"cuda:{rank:d}")

    generator: Union[DistributedDataParallel, Generator] = Generator(h).to(device)
    mpd: Union[DistributedDataParallel, MultiPeriodDiscriminator] = MultiPeriodDiscriminator().to(device)
    msd: Union[DistributedDataParallel, MultiScaleDiscriminator] = MultiScaleDiscriminator().to(device)

    if rank == 0:
        # print(generator)
        os.makedirs(arguments.checkpoint_path, exist_ok=True)
        # print("checkpoints directory : ", arguments.checkpoint_path)

    if os.path.isdir(arguments.checkpoint_path):
        cp_g = scan_checkpoint(arguments.checkpoint_path, "g_")
        cp_do = scan_checkpoint(arguments.checkpoint_path, "do_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=(h.adam_b1, h.adam_b2))
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=(h.adam_b1, h.adam_b2))

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = split_vctk_data(arguments.input_wavs_dir,
                                                             arguments.input_mels_dir)

    trainset = MelDataset(training_files=training_filelist,
                          base_mels_path=arguments.input_mels_dir,
                          segment_size=h.segment_size,
                          n_fft=h.n_fft,
                          num_mels=h.num_mels,
                          hop_size=h.hop_size,
                          win_size=h.win_size,
                          sampling_rate=h.sampling_rate,
                          fmin=h.fmin,
                          fmax=h.fmax,
                          fmax_loss=h.fmax_loss,
                          n_cache_reuse=0,
                          device=device,
                          fine_tuning=arguments.fine_tuning,
                          shuffle=h.num_gpus <= 1)

    train_sampler: Optional[DistributedSampler[int]] = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(training_files=validation_filelist,
                              base_mels_path=arguments.input_mels_dir,
                              segment_size=h.segment_size,
                              n_fft=h.n_fft,
                              num_mels=h.num_mels,
                              hop_size=h.hop_size,
                              win_size=h.win_size,
                              sampling_rate=h.sampling_rate,
                              fmin=h.fmin,
                              fmax=h.fmax,
                              fmax_loss=h.fmax_loss,
                              split=False,
                              shuffle=False,
                              fine_tuning=arguments.fine_tuning,
                              n_cache_reuse=0,
                              device=device)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(arguments.checkpoint_path, "logs"))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), arguments.training_epochs):
        if rank == 0:
            start = time.time()
            print(f"Epoch: {epoch + 1}")

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)  # type: ignore

        for _, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size,
                                          h.fmin, h.fmax_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % arguments.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print(f"Steps : {steps:d}, "
                          f"Gen Loss Total : {loss_gen_all:4.3f}, "
                          f"Mel-Spec. Error : {mel_error:4.3f}, "
                          f"s/b : {time.time() - start_b:4.3f}")

                # checkpointing
                if steps % arguments.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{arguments.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {"generator": (generator.module if h.num_gpus > 1 else generator).state_dict()}  # type: ignore
                    )
                    checkpoint_path = f"{arguments.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),  # type: ignore
                            "msd": (msd.module if h.num_gpus > 1 else msd).state_dict(),  # type: ignore
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch
                        }
                    )

                # Tensorboard summary logging
                if steps % arguments.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % arguments.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0.0
                    with torch.no_grad():
                        for j, batch_val in enumerate(validation_loader):
                            x, y, _, y_mel = batch_val
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(f"gt/y_{j}", y[0], steps, h.sampling_rate)
                                    sw.add_figure(f"gt/y_spec_{j}", plot_spectrogram(x[0]), steps)

                                sw.add_audio(f"generated/y_hat_{j}", y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure(f"generated/y_hat_spec_{j}",
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1  # noqa: SIM113

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")


def main() -> None:
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--group_name", default=None)
    parser.add_argument("--input_wavs_dir", default="LJSpeech-1.1/wavs")
    parser.add_argument("--input_mels_dir", default="ft_dataset")
    parser.add_argument("--input_training_file", default="LJSpeech-1.1/training.txt")
    parser.add_argument("--input_validation_file", default="LJSpeech-1.1/validation.txt")
    parser.add_argument("--checkpoint_path", default="cp_hifigan")
    parser.add_argument("--config", default="")
    parser.add_argument("--training_epochs", default=3100, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=5000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=1000, type=int)
    parser.add_argument("--fine_tuning", default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(**json_config)
    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f"Batch size per GPU: {h.batch_size}")
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h))
    else:
        mp.spawn(train, nprocs=1, args=(a, h,))
        # train(0, a, h)


if __name__ == "__main__":
    main()
