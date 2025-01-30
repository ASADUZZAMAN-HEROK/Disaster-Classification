import torch
import torch.nn as nn
from torchvision import models

from configs import Config


class ClassicModel(nn.Module):
    def __init__(self, in_channels: int, base_dim: int, num_classes: int):
        super(ClassicModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_dim, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(base_dim, base_dim * 2, 5)
        self.fc1 = nn.Linear(base_dim * 2 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x




def build_model(cfg: Config) -> ClassicModel:
    vgg16 = models.vgg16(weights=None)
    vgg16.load_state_dict(torch.load("../Dataset/vgg16-397923af.pth",weights_only=True))
    vgg16.classifier[6] = nn.Linear(4096, cfg.model.num_classes)
    for param in vgg16.features.parameters():
        param.requires_grad = False
    vgg16.classifier[6].requires_grad = True
    return vgg16
