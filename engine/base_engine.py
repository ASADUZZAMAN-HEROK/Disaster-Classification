"""
BaseTrainer is used to show the training details without making our final trainer too complicated.
Users can extend this class to add more functionalities.
"""

import os

import accelerate
import torch

from configs import Config
from utils.meter import AverageMeter
from utils.progress_bar import ProgressBars, tqdm_print


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.3f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


class BaseEngine:
    def __init__(
        self, accelerator: accelerate.Accelerator, cfg: Config, is_training_engine: bool = True
    ):
        # Setup accelerator for distributed training (or single GPU) automatically
        self.base_dir = os.path.join(cfg.log_dir, cfg.project_dir)
        self.accelerator = accelerator

        if self.accelerator.is_main_process and is_training_engine:
            os.makedirs(self.base_dir, exist_ok=True)
            tqdm_print(cfg.__str__())
        self.accelerator.wait_for_everyone()

        self.cfg = cfg
        self.device = self.accelerator.device
        tqdm_print(f'Using device: {self.device}')
        self.dtype = self.get_dtype()

        self.sub_task_progress = ProgressBars(leave=False, position=1)
        self.epoch_progress = ProgressBars(leave=True, position=0)

        # Monitor for the time
        self.iter_time = AverageMeter()
        self.data_time = AverageMeter()

    def get_dtype(self):
        if self.cfg.mixed_precision == "no":
            return torch.float32
        elif self.cfg.mixed_precision == "fp16":
            return torch.float16
        elif self.cfg.mixed_precision == "bf16":
            return torch.bfloat16

    def print_dataset_details(self):
        tqdm_print(
            "📁 Length of dataset\n"
            f" - 💪 Train: {len(self.train_loader.dataset)}\n"
            f" - 📝 Validation: {len(self.val_loader.dataset)}\n"
        
        )

    def print_model_details(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        total_params = trainable_params + non_trainable_params
        model_stats = f"🤖 Model Parameters:\n - 🔥 Trainable: {trainable_params}\n - 🧊 Non-trainable: {non_trainable_params}\n - 🧊 Total: {total_params}\n"
        tqdm_print(
           model_stats
        )

    def print_training_details(self):
        try:
            self.print_dataset_details()
        except Exception as e:
            tqdm_print("Error in printing dataset details:"+e)
        try:
            self.print_model_details()
        except Exception as e:
            tqdm_print("Error in printing model details"+e)

    def reset(self):
        self.data_time.reset()
        self.iter_time.reset()

    def close(self):
        self.accelerator.end_training()
