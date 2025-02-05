from __future__ import annotations

import dataclasses
from typing import List, Literal, Optional

from dataclass_wizard import JSONPyWizard
from configs.config_utils import str_config



@dataclasses.dataclass
class TrainingConfig(JSONPyWizard):
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
    def __str__(self):
        return str_config(self)


@dataclasses.dataclass
class EvalConfig(JSONPyWizard):
    num_workers: int = 4
    batch_size: int = 32
    def __str__(self):
        return str_config(self)


@dataclasses.dataclass
class ModelConfig(JSONPyWizard):
    name: str = "vgg16"
    in_channels: int = 3
    base_dim: int = 16
    num_classes: int = 10
    resume_path: Optional[str] = None
    pretrained: bool = False
    weight_path: Optional[str] = None
    def __str__(self):
        return str_config(self)


@dataclasses.dataclass
class DataConfig(JSONPyWizard):
    root: str = "data"
    train_dir: str = "train"
    test_dir: str = "test"
    train_val_split: float = 0.8
    def __str__(self):
        return str_config(self)


@dataclasses.dataclass
class Config(JSONPyWizard):
    config: str
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
    project_tracker: List[str] = None
    mixed_precision: str = "no"
    seed: int = 0
    
    def __str__(self):
        return str_config(self)
