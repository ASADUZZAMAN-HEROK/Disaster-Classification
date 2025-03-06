import json
import os
import time

import accelerate
import torch

from configs import Config
from dataset import get_train_loader
from engine.base_engine import BaseEngine
from modeling import build_loss, build_model
from utils.progress_bar import tqdm_print


class Engine(BaseEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)

        # Setup model, loss, optimizer, 
        model, input_transform, target_transform = build_model(cfg)
        self.loss_fn = build_loss(cfg)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.training.lr * self.accelerator.num_processes,
            weight_decay=self.cfg.training.weight_decay,
        )

        
        # Dataloaders
        with self.accelerator.main_process_first():
            train_loader, val_loader = get_train_loader(cfg, input_transform, target_transform)
        

        # Prepare model, optimizer, loss_fn, and dataloaders for distributed training (or single GPU)
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(model, optimizer, train_loader, val_loader)
        self.min_loss = float("inf")
        self.current_epoch = 1

        self.max_acc = 0
        self.sub_task = "Sub Task"
        self.main_task = "Main Task"

        # Resume or not
        if self.cfg.model.resume_path is not None:
            with self.accelerator.main_process_first():
                self.load_from_checkpoint()

    def load_from_checkpoint(self):
        """
        Load model and optimizer from checkpoint for resuming training.
        Modify this for custom components if needed.
        """
        checkpoint = self.cfg.model.resume_path
        if not os.path.exists(checkpoint):
            tqdm_print(f"[WARN] Checkpoint {checkpoint} not found. Skipping...")
            return
        self.accelerator.load_state(checkpoint)

        if not os.path.exists(os.path.join(checkpoint, "meta_data.json")):
            tqdm_print(
                f"[WARN] meta data for resuming training is not found in {checkpoint}. Skipping..."
            )
            return

        with open(os.path.join(checkpoint, "meta_data.json"), "r") as f:
            meta_data = json.load(f)
        self.current_epoch = meta_data.get("epoch", 0) + 1
        self.max_acc = meta_data.get("max_acc", 0)
        tqdm_print(
            f"[WARN] Checkpoint loaded from {self.cfg.model.resume_path}, continue training or validate..."
        )
        del checkpoint

    def save_checkpoint(self, save_path: str):
        self.accelerator.save_state(save_path)
        with open(os.path.join(save_path, "meta_data.json"), "w") as f:
            json.dump(
                {
                    "epoch": self.current_epoch,
                    "max_acc": self.max_acc,
                },
                f,
            )
    
    def save_model(self, save_path: str):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), save_path)

    def _train_one_epoch(self):
        self.progress_bar.add_task(self.sub_task, "Loader", total=len(self.train_loader))
        self.model.train()
        step_loss = 0
        total_acc = 0
        start = time.time()
        for loader_idx, (img, label) in enumerate(self.train_loader, 1):
            current_step = (self.current_epoch - 1) * len(self.train_loader) + loader_idx
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
                self.accelerator.log(
                    {
                        "loss/train": step_loss,
                    },
                    step=current_step,
                )
                step_loss = 0

            self.accelerator.log(
                {
                    "time/iter": self.iter_time.val,
                    "time/data": self.data_time.val,
                },
                step=current_step,
            )
            self.progress_bar.update(self.sub_task, advance=1)


            start = time.time()
        total_acc/= len(self.train_loader)
        tqdm_print(f"train. acc. at epoch {self.current_epoch}: {total_acc:.3f}")
        # self.progress_bar.remove_task(self.sub_task)
        return total_acc

    def validate(self):
        self.progress_bar.add_task(self.sub_task, "Validate", total=len(self.val_loader))
        total_acc = 0
        self.model.eval()
        for img, label in self.val_loader:
            pred = self.model(img)
            batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
            correct = (batch_pred.argmax(1) == batch_label).sum().item()
            total_acc += correct / len(label)
            self.progress_bar.update(self.sub_task, advance=1)
        total_acc /= len(self.val_loader)
        if self.accelerator.is_main_process:
            tqdm_print(f"val. acc. at epoch {self.current_epoch}: {total_acc:.3f}")
            self.accelerator.log(
                {
                    "acc/val": total_acc,
                },
                step=(self.current_epoch - 1) * len(self.train_loader),  # Use train steps
            )
        if self.accelerator.is_main_process and total_acc > self.max_acc:
            save_path = os.path.join(self.base_dir, "checkpoint")
            tqdm_print(f"new best found with: {total_acc:.3f}, save to {save_path}")
            self.save_model(os.path.join(self.base_dir, f"{self.cfg.model.name}_val_best.pth"))
            self.max_acc = total_acc
            self.save_checkpoint(
                os.path.join(
                    save_path,
                    f"epoch_{self.current_epoch}",
                ),
            )
        # self.progress_bar.remove_task(self.sub_task)
        return self.max_acc

    def setup_training(self):
        os.makedirs(os.path.join(self.base_dir, "checkpoint"), exist_ok=True)
        self.accelerator.init_trackers(
            self.accelerator.project_configuration.project_dir, config=self.cfg.to_dict()["training"]
        )


    def save_model_accuracy(self, model_name, train_accuracy, val_accuracy):
        file_path = "train_result.json"
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                data = json.load(file)
                data.update({model_name:{"train_accuracy": f'{train_accuracy:0.4f}', "val_accuracy":f'{val_accuracy:0.4f}'}})
                file.seek(0)
                file.truncate()
                json.dump(data, file, indent=4)
        else:
            with open(file_path, 'w') as file:
                json.dump({model_name:{"train_accuracy": f'{train_accuracy:0.4f}', "val_accuracy":f'{val_accuracy:0.4f}'}}, file, indent=4)

    def train(self):
        if self.accelerator.is_main_process:
            self.print_training_details()
            self.setup_training()
        self.accelerator.wait_for_everyone()
        self.progress_bar.add_task(
            self.main_task,
            desc="Training Epoch",
            total=self.cfg.training.epochs,
            completed=self.current_epoch - 1,
            acc=self.max_acc,
        )
        for epoch in range(self.current_epoch, self.cfg.training.epochs + 1):
            self.current_epoch = epoch
            train_accuracy = self._train_one_epoch()
            if epoch % self.cfg.training.val_freq == 0:
                self.accelerator.wait_for_everyone()
                val_accuracy = self.validate()
                if self.max_acc == val_accuracy:
                    self.save_model_accuracy(self.cfg.model.name, train_accuracy, val_accuracy)
            self.progress_bar.update(self.main_task, advance=1, acc=self.max_acc)
        self.progress_bar.stop_task(self.main_task)
        self.progress_bar.stop_task(self.sub_task)  

        self.accelerator.wait_for_everyone()
        self.save_model(os.path.join(self.base_dir, f"{self.cfg.model.name}_after_train.pth"))
    

    def reset(self):
        super().reset()
        self.max_acc = 0
