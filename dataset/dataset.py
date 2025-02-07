from typing import Tuple, List

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split

from configs import Config
from dataset.CDD_util import get_all_files, stratified_split
from PIL import Image
from collections import Counter
import os

    
class CDDFullDataset(Dataset):
    def __init__(self, cfg: Config, input_transform: transforms = None, output_transform: transforms = None):
        super().__init__()
        self.input_transform = input_transform
        self.target_transform = output_transform
        self.data_dir = cfg.data.root
        self.all_files = get_all_files(self.data_dir)
        self.all_labels = list(map(lambda x: x.split("/")[-2], self.all_files))
        unique_labels = sorted(set(map(lambda x: x.split("/")[-2], self.all_files)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        if self.target_transform is None:
            self.target_transform = lambda x: self.label_to_idx[x]

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.input_transform(image)
        except:
            print(f"Error reading image: {img_path}")
            print(img_path)
        label = self.all_labels[idx]
        label = self.target_transform(label)
        return image, label

    def get_all_labels(self):
        return self.all_labels
    
    def get_all_files(self):
        return self.all_files




def stratifiedSplitFromIdx(all_labels, idx, ratio, seed):
    class_idx = {}
    for i in idx:
        if all_labels[i] not in class_idx:
            class_idx[all_labels[i]]=[]
        class_idx[all_labels[i]].append(i)
    
    train_indices = []
    test_indices = []

    for label, i in class_idx.items():
        train, test = train_test_split(i, train_size=ratio, random_state=seed)
        train_indices.extend(train)
        test_indices.extend(test)

    
    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices

    



def get_kFold_dataloaders(cfg: Config, input_transform: transforms, output_transform: transforms) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:

    dataset = CDDFullDataset(cfg=cfg, input_transform=input_transform, output_transform=output_transform)
    all_files = dataset.get_all_files()
    all_labels = dataset.get_all_labels()
    stratified_kFold_split = StratifiedKFold(n_splits=cfg.training.num_fold, shuffle=True, random_state=cfg.seed)

    loaders = []
    for train_val_idx, test_idx in stratified_kFold_split.split(all_files, all_labels):
        train_idx, val_idx = stratifiedSplitFromIdx(all_labels, train_val_idx, cfg.data.train_val_split, cfg.seed)
        train_loader = DataLoader(
                Subset(dataset, train_idx),
                num_workers=cfg.training.num_workers,
                batch_size=cfg.training.batch_size,
                shuffle=True,
            )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            num_workers=cfg.evaluation.num_workers,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            Subset(dataset, test_idx),
            num_workers=cfg.evaluation.num_workers,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
        )
        print(f'Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}')
        loaders.append((train_loader, val_loader, test_loader))

    return loaders


