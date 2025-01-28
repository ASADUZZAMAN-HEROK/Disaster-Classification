from torchvision import models
import torch
import accelerate

vgg16 = models.vgg16()

vgg16.load_state_dict(torch.load("vgg16-397923af.pth", weights_only=True))
print(vgg16)
