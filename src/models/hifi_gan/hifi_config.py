from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HiFiGeneratorParam:
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_initial_channel: int = field(default=512)
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    resblock_initial_channel: int = field(default=256)
    resblock: str = field(default="1")


@dataclass
class TrainParamsHiFi:
    model_param: HiFiGeneratorParam
    early_stopping: int = field(default=10)
    fmax_loss: Optional[int] = field(default=None)
    batch_size: int = field(default=16)
    learning_rate: float = field(default=0.0002)
    adam_b1: float = field(default=0.8)
    adam_b2: float = field(default=0.99)
    lr_decay: float = field(default=0.999)
    segment_size: int = field(default=8192)
    training_epochs: int = field(default=1000)
    logging_interval: int = field(default=5)
    checkpoint_interval: int = field(default=5000)
    summary_interval: int = field(default=1000)
    fine_tuning: bool = field(default=True)
    split_data: bool = field(default=True)
