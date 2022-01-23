from dataclasses import dataclass, field

import yaml
from marshmallow_dataclass import class_schema

from src.constants import PATHLIKE
from src.data_process.config import VCTKDatasetParams
from src.models.feature_models.config import ModelParams
from src.models.hifi_gan import HIFIParams


@dataclass
class SchedulerParams:
    start_decay: int = field(default=4000)
    decay_steps: int = field(default=50000)
    decay_rate: float = field(default=0.5)
    last_epoch: int = field(default=400000)


@dataclass
class OptimizerParams:
    learning_rate: float = field(default=0.001)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-06)
    reg_weight: float = field(default=1e-06)


@dataclass
class LossParams:
    mels_weight: float = field(default=1.0)
    duration_weight: float = field(default=2.0)
    adversarial_weight: float = field(default=5e-3)


@dataclass
class TrainParams:
    data: VCTKDatasetParams
    test_size: float
    model: ModelParams
    optimizer: OptimizerParams
    scheduler: SchedulerParams
    loss: LossParams
    device: str
    checkpoint_name: str
    hifi: HIFIParams
    iters_per_checkpoint: int = field(default=10000)
    early_stopping_rounds: int = field(default=5)
    log_steps: int = field(default=1000)
    epochs: int = field(default=2500)
    batch_size: int = field(default=16)
    seed: int = field(default=42)
    sample_rate: int = field(default=22050)
    grad_clip_thresh: float = field(default=1.0)
    hop_size: int = field(default=256)
    f_min: int = field(default=0)
    f_max: int = field(default=8000)
    win_size: int = field(default=1024)
    n_fft: int = field(default=1024)
    n_mels: int = field(default=80)


TrainConfigSchema = class_schema(TrainParams)


def load_config(path: PATHLIKE) -> TrainParams:
    with open(path, "r") as input_stream:
        schema = TrainConfigSchema()
        return schema.load(yaml.safe_load(input_stream))
