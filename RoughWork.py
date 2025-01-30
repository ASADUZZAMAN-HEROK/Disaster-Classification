import random
import math
import torch.utils.data
from collections import defaultdict

import tyro
from torchvision.transforms import transforms
from configs.config import Config
from dataset.dataset import CDDDAtaset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

img = Image.open("../Dataset/ComprehensiveDisasterDataset/Human_Damage/02_0061.png").convert("RGB") 


train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
img = train_transform(img)
# exit()

cfg = tyro.cli(Config)
if cfg.config is not None:
    with open(cfg.config, "r") as f:
        json_cfg = Config.from_json(f.read())
    if cfg.model.resume_path is not None:
        json_cfg.model.resume_path = cfg.model.resume_path
    cfg = json_cfg 

fullDataset = CDDDAtaset(cfg)

dataloader = torch.utils.data.DataLoader(fullDataset, batch_size=1, shuffle=False, num_workers=1)

all_labels = fullDataset.get_labels()


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
    return first_set_inputs, second_set_inputs

train_dataset, val_dataset= stratified_split(fullDataset, all_labels, 0.8, cfg.seed)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Total dataset size: {len(fullDataset)}")