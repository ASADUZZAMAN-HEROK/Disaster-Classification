from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from configs import Config
from dataset.CDD_util import get_all_files, stratified_split
from PIL import Image
from collections import Counter
import os


class CDDDAtaset(Dataset):
    """Please define your own `Dataset` here. We provide an example for CIFAR-10 dataset."""
    def __init__(self, cfg: Config, is_train: bool = True, input_transform: transforms = None, output_transform: transforms = None):
        super().__init__()
        self.input_transform = input_transform
        self.target_transform = output_transform
        self.data_dir = os.path.join(cfg.data.root, cfg.data.train_dir if is_train else cfg.data.test_dir)
        print(self.data_dir)
        self.all_files = get_all_files(os.path.join(self.data_dir))  
        
        unique_labels = sorted(set(map(lambda x: x.split("/")[-2], self.all_files)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        if self.target_transform is None:
            self.target_transform = lambda x: self.label_to_idx[x]
        self.class_frequency = Counter(self.get_labels())
        print(self.class_frequency)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        label = img_path.split("/")[-2]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.input_transform(image)
        except:
            print(f"Error reading image: {img_path}")
            print(img_path)
        label = self.target_transform(label)
        return image, label
    def get_labels(self):
        return sorted(list(map(lambda x: x.split("/")[-2], self.all_files)))
    
    def get_all_files(self):
        return self.all_files



def get_train_loader(cfg: Config, input_transform: transforms, output_transform: transforms) -> Tuple[DataLoader, DataLoader]:
    fullDataset = CDDDAtaset(cfg, is_train=True, input_transform=input_transform, output_transform=output_transform)
    all_labels = fullDataset.get_labels()
    train_dataset, val_dataset, _, _ = stratified_split(fullDataset, all_labels, cfg.data.train_val_split, cfg.seed)
    del fullDataset
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader,

def get_test_loader(cfg: Config)->DataLoader:
    test_dataset = CDDDAtaset(cfg, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
    )
    return test_loader

