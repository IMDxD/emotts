from dataclasses import dataclass, field
from typing import List


@dataclass
class VCTKDatasetParams:

    text_dir: str
    mels_dir: str
    wav_dir: str
    feature_dir: str
    ignore_speakers: List[int]
    text_ext: str = field(default=".TextGrid")
    mels_ext: str = field(default=".pkl")

