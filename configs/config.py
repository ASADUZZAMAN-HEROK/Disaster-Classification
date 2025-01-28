from __future__ import annotations

import dataclasses
from typing import List, Literal, Optional

from dataclass_wizard import JSONPyWizard


@dataclasses.dataclass
class TrainingConfig:
    engine: str = "engine"
    label_smoothing: float = 0.0
    batch_size: int = 32
    val_freq: int = 1
    epochs: int = 50
    num_workers: int = 4
    accum_iter: int = 1
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    lr: float = 0.0003
    weight_decay: float = 0.0001


@dataclasses.dataclass
class EvalConfig:
    num_workers: int = 4
    batch_size: int = 32


@dataclasses.dataclass
class ModelConfig:
    in_channels: int = 3
    base_dim: int = 16
    num_classes: int = 10
    resume_path: Optional[str] = None


@dataclasses.dataclass
class DataConfig:
    root: str = "data"


@dataclasses.dataclass
class Config(JSONPyWizard):
    # Config for training option
    training: TrainingConfig

    # Config for model option
    model: ModelConfig

    # Config for data option
    data: DataConfig

    # Config for evaluation option
    evaluation: EvalConfig

    project_dir: str = "project"
    log_dir: str = "logs"
    project_tracker: List[str] = dataclasses.field(default_factory=lambda: ["tensorboard"])
    mixed_precision: str = "no"
    seed: int = 0
    config: Optional[str] = None
