import random
import math
import torch.utils.data
from collections import defaultdict

import tyro
from torchvision.transforms import transforms
from configs.config import Config
from dataset.CDD_util import get_all_files
from dataset.dataset import CDDDAtaset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from shutil import copyfile


cfg = tyro.cli(Config)
if cfg.config is not None:
    with open(cfg.config, "r") as f:
        json_cfg = Config.from_json(f.read())
    cfg = json_cfg 

fullDataset = CDDDAtaset(cfg)
# dataloader = torch.utils.data.DataLoader(fullDataset, batch_size=1, shuffle=False, num_workers=1)

all_labels = fullDataset.get_labels()
print(fullDataset.__len__())

for a, b in zip(fullDataset.class_frequency.items(), fullDataset.augmented_class_frequency.items()):
    print(a, b)


def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, random_state=None):
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    # first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    # second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, second_set_inputs, first_set_indices, second_set_indices

train_dataset, val_dataset, first_set_indices, second_set_indices =stratified_split(fullDataset, all_labels, 0.85, cfg.seed)
    
    
# base_path = "../Dataset/ComprehensiveDisasterDataset"    
# for idx in tqdm(second_set_indices):
#     img_path = fullDataset.all_files[idx]
#     rel_path = os.path.relpath(img_path, base_path)
#     dir = os.path.dirname(rel_path)
#     if not os.path.exists(f"../Dataset/CDD/Test/{dir}"):
#         os.makedirs(f"../Dataset/CDD/Test/{dir}")
#     copyfile(img_path, f"../Dataset/CDD/Test/{rel_path}")
    

# for idx in tqdm(first_set_indices):
#     img_path = fullDataset.all_files[idx]
#     rel_path = os.path.relpath(img_path, base_path)
#     dir = os.path.dirname(rel_path)
#     if not os.path.exists(f"../Dataset/CDD/Train/{dir}"):
#         os.makedirs(f"../Dataset/CDD/Train/{dir}")
#     copyfile(img_path, f"../Dataset/CDD/Train/{rel_path}")
    
# train_size = len(get_all_files("../Dataset/CDD/Train"))
# test_size = len(get_all_files("../Dataset/CDD/Test"))

# print(f"Train size: {train_size}")
# print(f"Test size: {test_size}")
