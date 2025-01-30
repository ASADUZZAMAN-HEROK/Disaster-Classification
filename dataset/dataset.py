from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from configs import Config
from dataset.CDD_util import get_all_files, stratified_split
from PIL import Image


class CDDDAtaset(Dataset):
    """Please define your own `Dataset` here. We provide an example for CIFAR-10 dataset."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.all_files = get_all_files(cfg.data.root)
        self.input_transform = transforms.Compose([
            transforms.Resize(256),                      # Resize image to 256px
            transforms.CenterCrop(224),                   # Crop the image to 224x224
            transforms.ToTensor(),                        # Convert image to PyTorch tensor
            transforms.Normalize(                         # Normalize using ImageNet mean and std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        unique_labels = set(map(lambda x: x.split("/")[-2], self.all_files))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.target_transform = lambda x: self.label_to_idx[x]
    
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
        return list(map(lambda x: x.split("/")[-2], self.all_files))



def get_loader(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    fullDataset = CDDDAtaset(cfg)
    all_labels = fullDataset.get_labels()
    train_dataset, test_dataset, train_labels, _ = stratified_split(fullDataset, all_labels, 0.91, cfg.seed)
    train_dataset, val_dataset, _, _ = stratified_split(train_dataset, train_labels, 0.9, cfg.seed)
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
    
    test_dataset = DataLoader(
        test_dataset,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_dataset
