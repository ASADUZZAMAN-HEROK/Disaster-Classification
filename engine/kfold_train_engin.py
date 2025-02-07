import json
import os
import time

import accelerate
import torch

from configs import Config
from dataset import get_kFold_dataloaders
from engine.base_engine import BaseEngine
from modeling import build_loss, build_model
from tqdm import tqdm
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class KFoldEngine(BaseEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)

        _, input_transform, target_transform = build_model(self.cfg)
        with self.accelerator.main_process_first():
            self.loaders = get_kFold_dataloaders(cfg, input_transform, target_transform)

        self.fold_max_acc = 0
    
    def fresh_model(self, train_loader, val_loader, test_loader):
        model, _, _ = build_model(self.cfg)
        self.loss_fn = build_loss(self.cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.training.lr * self.accelerator.num_processes,
            weight_decay=self.cfg.training.weight_decay,
        )
        (
            self.model,
            self.optimizer,
            self.train_loader, 
            self.val_loader, 
            self.test_loader
         ) = self.accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)
        self.min_loss = float("inf")

    def reset_and_load_best(self):
        self.model.load_state_dict(torch.load(f"Logs/{self.cfg.project_dir}/{self.cfg.model.name}_val_best.pth", map_location=self.accelerator.device, weights_only=True))  
        

    def save_model(self, save_path: str):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), save_path)

    def _train_one_epoch(self, train_loader):
        epoch_progress = self.sub_task_progress.add_task("loader", total=len(train_loader))
        self.model.train()
        step_loss = 0
        total_acc = 0
        start = time.time()
        for loader_idx, (img, label) in enumerate(train_loader, 1):
            self.data_time.update(time.time() - start)
            with self.accelerator.accumulate(self.model):
                output = self.model(img)
                loss = self.loss_fn(output, label)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss = self.accelerator.gather(loss.detach().cpu().clone()).mean()
                step_loss += loss.item() / self.cfg.training.accum_iter

                batch_pred, batch_label = self.accelerator.gather_for_metrics((output, label))
                correct = (batch_pred.argmax(1) == batch_label).sum().item()
                total_acc += correct /len(label)

            self.iter_time.update(time.time() - start)

            if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                step_loss = 0
            self.sub_task_progress.update(epoch_progress, advance=1)

            start = time.time()
        total_acc/= len(self.train_loader)
        # self.accelerator.print(f"train. acc. now: {total_acc:.3f}")
        self.sub_task_progress.remove_task(epoch_progress)
        return total_acc

    def validate(self, val_loader):
        valid_progress = self.sub_task_progress.add_task("validate", total=len(self.val_loader))
        total_acc = 0
        self.model.eval()
        for img, label in val_loader:
            pred = self.model(img)
            batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
            correct = (batch_pred.argmax(1) == batch_label).sum().item()
            total_acc += correct / len(label)
            self.sub_task_progress.update(valid_progress, advance=1)
        total_acc /= len(self.val_loader)
        # if self.accelerator.is_main_process:
        #     self.accelerator.print(f"val. acc. at epoch now: {total_acc:.3f}")
    
        if self.accelerator.is_main_process and total_acc > self.max_val_acc:
            self.accelerator.print(f"new best found with: {total_acc:.3f}")
            self.save_model(os.path.join(self.base_dir, f"{self.cfg.model.name}_val_best.pth"))
            self.max_val_acc = total_acc

        self.sub_task_progress.remove_task(valid_progress)
        return total_acc
    
    def test(self, test_loader):
        valid_progress = self.sub_task_progress.add_task("Testing", total=len(self.test_loader))
        total_acc = 0
        self.model.eval()
        for img, label in test_loader:
            pred = self.model(img)
            batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
            correct = (batch_pred.argmax(1) == batch_label).sum().item()
            total_acc += correct / len(label)
            self.sub_task_progress.update(valid_progress, advance=1)
        total_acc /= len(test_loader)
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Test. acc. now : {total_acc:.3f}")

        if self.accelerator.is_main_process and total_acc > self.fold_max_acc:
            self.fold_max_acc = total_acc

        self.sub_task_progress.remove_task(valid_progress)
        return total_acc

    def setup_training(self):
        os.makedirs(os.path.join(self.base_dir, "checkpoint"), exist_ok=True)
        self.accelerator.init_trackers(
            self.accelerator.project_configuration.project_dir, config=self.cfg.to_dict()["training"]
        )

    def save_model_accuracy(self, model_name, train_accuracy, val_accuracy, test_accuracy):
        file_path = "kfold_result.json"
        try:
            with open(file_path, 'r+') as file:
                data = json.load(file)
                data.update({model_name:{"train_accuracy": f'{train_accuracy:0.4f}', "val_accuracy":f'{val_accuracy:0.4f}', "test_accuracy":f'{test_accuracy:0.4f}'}})
                file.seek(0)
                file.truncate()
                json.dump(data, file, indent=4)
        except:
            with open(file_path, 'w') as file:
                json.dump({model_name:{"train_accuracy": f'{train_accuracy:0.4f}', "val_accuracy":f'{val_accuracy:0.4f}', "test_accuracy":f'{test_accuracy:0.4f}'}},file)

    def train(self, train_loader, val_loader):
        train_progress = self.epoch_progress.add_task(
            "Epoch",
            total=self.cfg.training.epochs,
            completed = 0,
            acc=self.max_val_acc,
        )
        self.accelerator.wait_for_everyone()
        for epoch in range(self.cfg.training.epochs):
            train_accuracy = self._train_one_epoch(train_loader=train_loader)
            if epoch % self.cfg.training.val_freq == 0:
                self.accelerator.wait_for_everyone()
                val_accuracy = self.validate(val_loader=val_loader)
                if self.max_val_acc == val_accuracy:
                    self.train_acc_max_val = train_accuracy
                
            self.epoch_progress.update(train_progress, advance=1, acc=self.max_val_acc)
        self.epoch_progress.stop_task(train_progress)
        self.accelerator.wait_for_everyone()
    
    def kFold_train(self):
        fold_progress = self.fold_progress.add_task(description="K Fold Train", total=self.cfg.training.num_fold,
            completed=0,
            acc=0,
        )
        if self.accelerator.is_main_process:
            # self.print_training_details()
            self.setup_training()
        self.accelerator.wait_for_everyone()
        for i, (train_loader, val_loader, test_loader) in enumerate(self.loaders, 1):
            self.reset()
            self.fresh_model(train_loader, val_loader, test_loader)
            self.train(train_loader=self.train_loader, val_loader=self.val_loader)
            self.reset_and_load_best()
            fold_acc = self.test(self.test_loader)
            self.save_model_accuracy(f'{self.cfg.model.name}/fold_{i}', self.train_acc_max_val, self.max_val_acc, fold_acc)
            self.fold_progress.update(fold_progress,advance=1,acc=fold_acc)
        self.fold_progress.stop_task(fold_progress)
        self.accelerator.wait_for_everyone()

    def reset(self):
        super().reset()
        self.max_val_acc = 0
        self.train_acc_max_val = 0
