import json
import os
import time

import accelerate
import torch

from configs import Config
from dataset import get_test_loader
from engine.base_engine import BaseEngine
from modeling import build_loss, build_model
from tqdm import tqdm


class TestEngine(BaseEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)

        model, input_transform, target_transform = build_model(cfg)
        self.loss_fn = build_loss(cfg)

        # Dataloaders
        with self.accelerator.main_process_first():
            test_loader = get_test_loader(cfg, input_transform, target_transform)
        

        # Prepare model, optimizer, loss_fn, and dataloaders for distributed training (or single GPU)
        (
            self.model,
            self.test_loader,
        ) = self.accelerator.prepare(model,test_loader)
        self.min_loss = float("inf")
        
        self.accelerator.print(
            "📁 \033[1mLength of dataset\033[0m:\n"
            f" - 📝 Test: {len(self.test_loader.dataset)}\n"
        )

    def batch_test(self):
        if self.accelerator.is_main_process:
            self.setup_test()
        print("Start testing")
        test_progress = self.sub_task_progress.add_task("test", total=len(self.test_loader))
        total_acc = 0

        self.model.eval()
        self.accelerator.print("Model is in evaluation mode")
        for img, label in self.test_loader  :
            pred = self.model(img)
            batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
            correct = (batch_pred.argmax(1) == batch_label).sum().item()
            total_acc += correct / len(label)
            self.sub_task_progress.update(test_progress, advance=1)
        total_acc /= len(self.test_loader)
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Average Test acc: {total_acc:.3f}")
            self.save_model_accuracy(self.cfg.model.name, total_acc)
        self.sub_task_progress.stop_task(test_progress)
        self.accelerator.wait_for_everyone()

    def setup_test(self):
        self.accelerator.init_trackers(
            self.accelerator.project_configuration.project_dir, config=self.cfg.to_dict()["evaluation"]
        )
    
    def save_model_accuracy(self, model_name, accuracy):
        file_path = "test_result.json"
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                data = json.load(file)
                data.update({model_name:round(accuracy, 4)})
                file.seek(0)
                file.truncate()
                json.dump(data, file, indent=4)
        else:
            with open(file_path, 'w') as file:
                json.dump({model_name:round(accuracy, 4)}, file, indent=4)
    

    def reset(self):
        super().reset()
        self.max_acc = 0
