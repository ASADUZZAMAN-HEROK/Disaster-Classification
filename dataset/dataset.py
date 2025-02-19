from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from configs import Config
from dataset.CDD_util import get_all_files, stratified_split
from PIL import Image
from collections import Counter
import os


import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter
import numpy as np

class CDDDAtaset(Dataset):
    """Custom Dataset with oversampling and augmentation for minority classes."""
    def __init__(self, cfg: Config, is_train: bool = True, input_transform: transforms = None, output_transform: transforms = None):
        super().__init__()
        self.input_transform = input_transform
        self.target_transform = output_transform
        self.data_dir = os.path.join(cfg.data.root, cfg.data.train_dir if is_train else cfg.data.test_dir)
        self.all_files = self.get_all_files()  
        
        # Map labels to indices
        unique_labels = sorted(set(map(lambda x: x.split("/")[-2], self.all_files)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Set target transform if not provided
        if self.target_transform is None:
            self.target_transform = lambda x: self.label_to_idx[x]
        
        # Calculate class frequencies
        self.class_frequency = Counter(self.get_labels())
        
        # Identify minority classes (e.g., classes with frequency below a threshold)
        self.minority_classes = self.identify_minority_classes(threshold=0.1 * len(self.all_files))
        
        # Define augmentation transforms for minority classes
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
        ])
        
        # Oversample minority classes
        self.oversampled_files = self.oversample_minority_classes()
        self.augmented_class_frequency = Counter(self.get_augmented_labels())
    
    def __len__(self):
        return len(self.oversampled_files)
    
    def __getitem__(self, idx):
        img_path, is_minority = self.oversampled_files[idx]
        label = img_path.split("/")[-2]
        label_idx = self.target_transform(label)
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Apply augmentation if the sample is from a minority class
            if is_minority:
                image = self.augmentation_transforms(image)
            elif self.input_transform:
                image = self.input_transform(image)
        except Exception as e:
            print(f"Error reading image: {img_path}")
            print(e)
            return None, None
        
        return image, label_idx
    
    def get_labels(self):
        """Get all labels from the dataset."""
        return list(map(lambda x: x.split("/")[-2], self.all_files))

    def get_augmented_labels(self):
        return list(map(lambda x: x[0].split("/")[-2], self.oversampled_files))
    
    def get_all_files(self):
        """Get all file paths in the dataset."""
        all_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    all_files.append(os.path.join(root, file))
        return all_files
    
    def identify_minority_classes(self, threshold):
        """Identify minority classes based on a frequency threshold."""
        minority_classes = []
        for label, count in self.class_frequency.items():
            if count < threshold:
                minority_classes.append(self.label_to_idx[label])
        return minority_classes
    
    def oversample_minority_classes(self):
        """Oversample minority classes to balance the dataset."""
        oversampled_files = []
        
        # Determine the maximum class frequency
        max_frequency = 3000 #max(self.class_frequency.values())
        
        for file in self.all_files:
            label = file.split("/")[-2]
            label_idx = self.label_to_idx[label]
            
            # Add the original sample
            oversampled_files.append((file, False))
            
            # If the class is a minority class, oversample it
            if label_idx in self.minority_classes:
                oversampling_factor = max_frequency // self.class_frequency[label]
                for _ in range(oversampling_factor - 1):
                    oversampled_files.append((file, True))  # Mark as minority for augmentation
        
        return oversampled_files



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

def get_test_loader(cfg: Config, input_transform: transforms, output_transform: transforms)->DataLoader:
    test_dataset = CDDDAtaset(cfg, is_train=False, input_transform=input_transform, output_transform=output_transform)
    test_loader = DataLoader(
        test_dataset,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
    )
    return test_loader

