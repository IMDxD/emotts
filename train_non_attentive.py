import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.constants import CHECKPOINT_DIR
from src.data_process import VctkCollate, VctkDataset
from src.models import NonAttentiveTacotron, NonAttentiveTacotronLoss
from src.train_config import TrainParams, load_config


def prepare_dataloaders(config: TrainParams) -> Tuple[DataLoader, DataLoader]:
    # Get data, data loaders and collate function ready
    trainset = VctkDataset(
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        config=config.train_data,
    )
    valset = VctkDataset(
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        config=config.valid_data,
    )
    collate_fn = VctkCollate()

    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=config.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        valset,
        sampler=None,
        num_workers=1,
        shuffle=False,
        batch_size=config.batch_size,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def load_model(config: TrainParams, n_speakers: int) -> NonAttentiveTacotron:
    model = NonAttentiveTacotron(
        n_mel_channels=config.n_mels,
        n_phonems=config.n_phonemes,
        n_speakers=n_speakers,
        device=torch.device(config.device),
        config=config.model,
    )
    return model


def load_checkpoint(
    checkpoint_path: Path,
    model: NonAttentiveTacotron,
    optimizer: Adam,
    scheduler: StepLR,
) -> [NonAttentiveTacotron, Adam, StepLR]:
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scheduler.load_state_dict(checkpoint_dict['scheduler'])
    return model, optimizer, scheduler


def save_checkpoint(
    filepath: Path, model: NonAttentiveTacotron, optimizer: Adam, scheduler: StepLR
):
    torch.save(
        {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        },
        filepath,
    )


def validate(
    model: NonAttentiveTacotron,
    criterion: NonAttentiveTacotronLoss,
    val_loader: DataLoader,
    iteration: int,
) -> None:

    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            durations, mel_outputs_postnet, mel_outputs = model(batch)
            loss = criterion(
                mel_outputs, mel_outputs_postnet, durations, batch[3], batch[4]
            )
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))


def train(config: TrainParams):
    """Training and validation logging results to tensorboard and stdout
    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name

    train_loader, val_loader = prepare_dataloaders(config)
    model = load_model(config, len(train_loader.dataset.speaker_to_idx))

    optimizer_config = config.optimizer
    optimizer = Adam(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.reg_weight,
        betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        eps=optimizer_config.adam_epsilon,
    )

    scheduler = StepLR(
        optimizer=optimizer,
        step_size=config.scheduler.decay_steps,
        gamma=config.scheduler.decay_rate,
        last_epoch=config.scheduler.last_epoch,
    )

    criterion = NonAttentiveTacotronLoss(
        mels_weight=config.loss.mels_weight, duration_weight=config.loss.duration_weight
    )

    iteration = 0
    epoch_offset = 0
    if os.path.isfile(checkpoint_path):
        model, optimizer, scheduler = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )

    model.train()

    for epoch in range(epoch_offset, config.epochs):
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()

            optimizer.zero_grad()
            durations, mel_outputs_postnet, mel_outputs = model(batch)

            loss = criterion(
                mel_outputs, mel_outputs_postnet, durations, batch[3], batch[4]
            )
            reduced_loss = loss.item()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_thresh
            )

            optimizer.step()

            if (i + 1) % config.log_steps == 0:
                duration = time.perf_counter() - start
                print(
                    "Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss, grad_norm, duration
                    )
                )

            if (i + 1) % config.iters_per_checkpoint == 0:
                validate(model, criterion, val_loader, iteration)
                save_checkpoint(checkpoint_path, model, optimizer, scheduler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='configuration file path'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
