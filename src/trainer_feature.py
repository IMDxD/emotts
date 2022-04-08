import json
import os
from pathlib import Path
from typing import Dict, Optional, OrderedDict, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import (
    CHECKPOINT_DIR,
    FEATURE_CHECKPOINT_NAME,
    FEATURE_MODEL_FILENAME,
    GENERATED_PHONEMES,
    LOG_DIR,
    MELS_MEAN_FILENAME,
    MELS_STD_FILENAME,
    PHONEMES_FILENAME,
    REFERENCE_PATH,
    SPEAKERS_FILENAME,
)
from src.data_process import VCTKBatch, VCTKCollate, VCTKDataset, VCTKFactory
from src.models.feature_models import NonAttentiveTacotron
from src.models.feature_models.loss_function import NonAttentiveTacotronLoss
from src.models.hifi_gan.models import Generator, load_model as load_hifi
from src.train_config import TrainParams, load_config


class Trainer:

    OPTIMIZER_FILENAME = "optimizer.pth"
    ITERATION_FILENAME = "iter.json"
    ITERATION_NAME = "iteration"
    EPOCH_NAME = "epoch"
    SAMPLE_SIZE = 10

    def __init__(self, config_path: Path):
        self.config: TrainParams = load_config(config_path)
        base_model_path = Path(self.config.base_model)
        self.checkpoint_path = (
            CHECKPOINT_DIR / self.config.checkpoint_name / FEATURE_CHECKPOINT_NAME
        )
        mapping_folder = (
            self.checkpoint_path if self.config.finetune else base_model_path.parent
        )
        self.log_dir = LOG_DIR / self.config.checkpoint_name / FEATURE_CHECKPOINT_NAME
        self.references = list(REFERENCE_PATH.rglob("*.pkl"))
        self.create_dirs()
        self.phonemes_to_id: Dict[str, int] = {}
        self.speakers_to_id: Dict[str, int] = {}
        self.device = torch.device(self.config.device)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.iteration_step = 1
        self.upload_mapping(mapping_folder)
        self.train_loader, self.valid_loader = self.prepare_loaders()

        self.feature_model = NonAttentiveTacotron(
            n_mel_channels=self.config.n_mels,
            n_phonems=len(self.phonemes_to_id),
            n_speakers=len(self.speakers_to_id),
            config=self.config.model,
            finetune=self.config.finetune,
        ).to(self.device)

        if self.config.finetune:
            base_model_state = torch.load(base_model_path, map_location=self.device)
            self.feature_model.load_state_dict(base_model_state)

        self.style_fc = nn.Sequential(
            nn.Linear(self.config.model.gst_config.emb_dim, len(self.speakers_to_id)),
            nn.Softmax(),
        )
        self.style_fc = self.style_fc.to(self.device)

        self.vocoder: Generator = load_hifi(
            model_path=self.config.pretrained_hifi,
            hifi_config=self.config.train_hifi.model_param,
            num_mels=self.config.n_mels,
            device=self.device,
        )
        self.model_optimizer = Adam(
            self.feature_model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.reg_weight,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )
        self.discriminator_optimizer = Adam(
            self.style_fc.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.reg_weight,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )
        self.scheduler = StepLR(
            optimizer=self.model_optimizer,
            step_size=self.config.scheduler.decay_steps,
            gamma=self.config.scheduler.decay_rate,
        )
        self.criterion = NonAttentiveTacotronLoss(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            config=self.config.loss,
        )
        self.adversatial_criterion = nn.NLLLoss()

        self.upload_checkpoints()

    def batch_to_device(self, batch: VCTKBatch) -> VCTKBatch:
        batch_on_device = VCTKBatch(
            phonemes=batch.phonemes.to(self.device).detach(),
            num_phonemes=batch.num_phonemes.detach(),
            speaker_ids=batch.speaker_ids.to(self.device).detach(),
            durations=batch.durations.to(self.device).detach(),
            mels=batch.mels.to(self.device).detach(),
        )
        return batch_on_device

    def create_dirs(self) -> None:
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def mapping_is_exist(self) -> bool:
        if not os.path.isfile(self.checkpoint_path / SPEAKERS_FILENAME):
            return False
        if not os.path.isfile(self.checkpoint_path / PHONEMES_FILENAME):
            return False
        return True

    def get_last_model(self) -> Optional[Path]:
        models = list(self.checkpoint_path.rglob(f"*_{FEATURE_MODEL_FILENAME}"))
        if len(models) == 0:
            return None
        return max(models, key=lambda x: int(x.name.split("_")[0]))

    def checkpoint_is_exist(self) -> bool:  # noqa: CFQ004
        model_path = self.get_last_model()
        if model_path is None:
            return False
        if not (self.checkpoint_path / self.OPTIMIZER_FILENAME).is_file():
            return False
        if not (self.checkpoint_path / self.ITERATION_FILENAME).is_file():
            return False
        else:
            with open(self.checkpoint_path / self.ITERATION_FILENAME) as f:
                iter_dict = json.load(f)
                if (
                    self.EPOCH_NAME not in iter_dict
                    or self.ITERATION_NAME not in iter_dict
                ):
                    return False
        return True

    def upload_mapping(self, mapping_folder: Path) -> None:
        if self.mapping_is_exist():
            with open(mapping_folder / SPEAKERS_FILENAME) as f:
                self.speakers_to_id.update(json.load(f))
            with open(mapping_folder / PHONEMES_FILENAME) as f:
                self.phonemes_to_id.update(json.load(f))

    def upload_checkpoints(self) -> None:
        if self.checkpoint_is_exist():
            model_path = self.get_last_model()
            feature_model: NonAttentiveTacotron = torch.load(
                model_path, map_location=self.device
            )
            optimizer_state_dict: OrderedDict[str, torch.Tensor] = torch.load(
                self.checkpoint_path / self.OPTIMIZER_FILENAME, map_location=self.device
            )
            with open(self.checkpoint_path / self.ITERATION_FILENAME) as f:
                iteration_dict: Dict[str, int] = json.load(f)
            self.feature_model = feature_model
            self.model_optimizer.load_state_dict(optimizer_state_dict)
            self.scheduler = StepLR(
                optimizer=self.model_optimizer,
                step_size=self.config.scheduler.decay_steps,
                gamma=self.config.scheduler.decay_rate,
            )
            self.iteration_step = iteration_dict[self.ITERATION_NAME]

    def save_checkpoint(self) -> None:
        with open(self.checkpoint_path / SPEAKERS_FILENAME, "w") as f:
            json.dump(self.speakers_to_id, f)
        with open(self.checkpoint_path / PHONEMES_FILENAME, "w") as f:
            json.dump(self.phonemes_to_id, f)
        with open(self.checkpoint_path / self.ITERATION_FILENAME, "w") as f:
            json.dump({self.ITERATION_NAME: self.iteration_step}, f)
        torch.save(
            self.feature_model,
            self.checkpoint_path / f"{self.iteration_step}_{FEATURE_MODEL_FILENAME}",
        )
        torch.save(
            self.model_optimizer.state_dict(),
            self.checkpoint_path / self.OPTIMIZER_FILENAME,
        )
        torch.save(
            self.train_loader.dataset.mels_mean,
            self.checkpoint_path / MELS_MEAN_FILENAME,
        )
        torch.save(
            self.train_loader.dataset.mels_std, self.checkpoint_path / MELS_STD_FILENAME
        )

    def prepare_loaders(self) -> Tuple[DataLoader[VCTKBatch], DataLoader[VCTKBatch]]:

        factory = VCTKFactory(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            n_mels=self.config.n_mels,
            config=self.config.data,
            phonemes_to_id=self.phonemes_to_id,
            speakers_to_id=self.speakers_to_id,
            ignore_speakers=self.config.data.ignore_speakers,
            finetune=self.config.finetune,
        )
        self.phonemes_to_id = factory.phoneme_to_id
        self.speakers_to_id = factory.speaker_to_id
        trainset, valset = factory.split_train_valid(self.config.test_size)
        collate_fn = VCTKCollate()

        train_loader = DataLoader(
            trainset,
            shuffle=False,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            valset,
            shuffle=False,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader  # type: ignore

    def write_losses(
        self,
        tag: str,
        total_loss: float,
        prenet_loss: float,
        postnet_loss: float,
        durations_loss: float,
    ) -> None:
        self.writer.add_scalar(
            f"Loss/{tag}/total", total_loss, global_step=self.iteration_step
        )
        self.writer.add_scalar(
            f"Loss/{tag}/prenet", prenet_loss, global_step=self.iteration_step
        )
        self.writer.add_scalar(
            f"Loss/{tag}/postnet", postnet_loss, global_step=self.iteration_step
        )
        self.writer.add_scalar(
            f"Loss/{tag}/durations", durations_loss, global_step=self.iteration_step
        )

    def vocoder_inference(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = tensor.unsqueeze(0).to(self.device)
            y_g_hat = self.vocoder(x)
            audio = y_g_hat.squeeze()
        return audio

    def calc_gst_loss(
        self, style_emb: torch.Tensor, batch: VCTKBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        discrim_output = self.style_fc(style_emb)

        log_adv_model = torch.log(1 - discrim_output)
        log_adv_dicriminator = torch.log(discrim_output)
        loss_adv_model = (
            self.config.loss.adversarial_weight
            * self.adversatial_criterion(log_adv_model, batch.speaker_ids)
        )
        loss_adv_discriminator = (
            self.config.loss.adversarial_weight
            * self.adversatial_criterion(log_adv_dicriminator, batch.speaker_ids)
        )
        return loss_adv_model, loss_adv_discriminator

    def train(self) -> None:
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.feature_model.train()

        while self.iteration_step < self.config.total_iterations:
            for batch in self.train_loader:
                batch = self.batch_to_device(batch)
                self.model_optimizer.zero_grad()
                (
                    durations,
                    mel_outputs_postnet,
                    mel_outputs,
                    style_emb,
                ) = self.feature_model(batch)

                loss_prenet, loss_postnet, loss_durations = self.criterion(
                    mel_outputs,
                    mel_outputs_postnet,
                    durations,
                    batch.durations,
                    batch.mels,
                )
                loss_adv_model, loss_adv_discriminator = self.calc_gst_loss(
                    style_emb, batch
                )
                loss = loss_prenet + loss_postnet + loss_durations
                loss_full = loss + loss_adv_model
                loss_full.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.feature_model.parameters(), self.config.grad_clip_thresh
                )

                self.model_optimizer.step()

                self.discriminator_optimizer.zero_grad()

                loss_adv_discriminator.backward()

                self.discriminator_optimizer.step()

                if (
                    self.config.scheduler.start_decay
                    <= self.iteration_step
                    <= self.config.scheduler.last_epoch
                ):
                    self.scheduler.step()

                if self.iteration_step % self.config.log_steps == 0:
                    self.write_losses(
                        "train", loss, loss_prenet, loss_postnet, loss_durations,
                    )

                if self.iteration_step % self.config.iters_per_checkpoint == 0:
                    self.feature_model.eval()
                    self.validate()
                    self.generate_samples()
                    self.save_checkpoint()
                    self.feature_model.train()

                self.iteration_step += 1
                if self.iteration_step >= self.config.total_iterations:
                    break

        self.writer.close()

    def validate(self) -> float:
        with torch.no_grad():
            val_loss = 0.0
            val_loss_prenet = 0.0
            val_loss_postnet = 0.0
            val_loss_durations = 0.0
            for batch in self.valid_loader:
                batch = self.batch_to_device(batch)
                durations, mel_outputs_postnet, mel_outputs, _ = self.feature_model(
                    batch
                )
                loss_prenet, loss_postnet, loss_durations = self.criterion(
                    mel_outputs,
                    mel_outputs_postnet,
                    durations,
                    batch.durations,
                    batch.mels,
                )
                loss = loss_prenet + loss_postnet + loss_durations
                val_loss += loss.item()
                val_loss_prenet += loss_prenet.item()
                val_loss_postnet += loss_postnet.item()
                val_loss_durations += loss_durations.item()

            val_loss = val_loss / len(self.valid_loader)
            val_loss_prenet = val_loss_prenet / len(self.valid_loader)
            val_loss_postnet = val_loss_postnet / len(self.valid_loader)
            val_loss_durations = val_loss_durations / len(self.valid_loader)
            self.write_losses(
                "valid",
                val_loss,
                val_loss_prenet,
                val_loss_postnet,
                val_loss_durations,
            )

    def generate_samples(self) -> None:
        valid_dataset: VCTKDataset = self.valid_loader.dataset  # type: ignore
        mels_mean = valid_dataset.mels_mean.float()
        mels_std = valid_dataset.mels_std.float()
        phonemes = [
            [self.phonemes_to_id.get(p, 0) for p in sequence]
            for sequence in GENERATED_PHONEMES
        ]
        audio_folder = self.checkpoint_path / f"{self.iteration_step}"
        audio_folder.mkdir(exist_ok=True, parents=True)
        with torch.no_grad():

            for reference_path in self.references:
                for i, sequence in enumerate(phonemes):
                    phonemes_tensor = torch.LongTensor([sequence]).to(self.device)
                    num_phonemes_tensor = torch.IntTensor([len(sequence)])
                    speaker = reference_path.parent.name
                    speaker_id = self.speakers_to_id[speaker]
                    reference = (torch.load(reference_path) - mels_mean) / mels_std
                    batch = (
                        phonemes_tensor,
                        num_phonemes_tensor,
                        torch.LongTensor([speaker_id]).to(self.device),
                        reference.to(self.device).permute(0, 2, 1),
                    )
                    output = self.feature_model.inference(batch)
                    output = output.permute(0, 2, 1).squeeze(0)
                    output = output * mels_std.to(self.device) + mels_mean.to(
                        self.device
                    )
                    audio = self.vocoder_inference(output.float())

                    name = f"{speaker}_{reference_path.stem}_{i}"
                    self.writer.add_audio(
                        f"Audio/Val/{name}",
                        audio.cpu(),
                        sample_rate=self.config.sample_rate,
                        global_step=self.iteration_step,
                    )
