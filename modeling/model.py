import torch
import torch.nn as nn
from torchvision import models, transforms

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



def build_vgg16(cfg: Config):
    vgg16 = models.vgg16()
    
    if cfg.model.pretrained:
        vgg16 = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    else:
        vgg16 = models.vgg16(weights=None)
    
    vgg16.classifier[6] = nn.Linear(4096, cfg.model.num_classes)
    for param in vgg16.features.parameters():
        param.requires_grad = not cfg.model.pretrained
    for i in range(6):
        vgg16.classifier[i].requires_grad = True

    if cfg.model.weight_path:
        vgg16.load_state_dict(torch.load(cfg.model.weight_path,weights_only=True))
    
    vgg16_transforms = transforms.Compose([
        transforms.Resize((256,256)),               # Resize the smaller edge to 256
        transforms.CenterCrop(224),            # Crop the center to 224x224
        transforms.ToTensor(),                 # Convert image to Tensor
        transforms.Normalize(                  # Normalize the image
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return vgg16, vgg16_transforms, None

def build_resnet50(cfg: Config):
    resnet50 = models.resnet50()
    
    if cfg.model.pretrained:
        resnet50 = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet50 = models.resnet50(weights=None)
    
    resnet50.fc = nn.Linear(2048, cfg.model.num_classes)
    for param in resnet50.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in resnet50.fc.parameters():
        param.requires_grad = True
    
    if cfg.model.weight_path:
        resnet50.load_state_dict(torch.load(cfg.model.weight_path,weights_only=True))

    resnet50_transforms = transforms.Compose([
        transforms.Resize((256,256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return resnet50, resnet50_transforms, None


def build_inceptionV3(cfg: Config):
    
    if cfg.model.pretrained:
        inceptionV3 = models.inception_v3(weights=models.inception.Inception_V3_Weights.IMAGENET1K_V1)
    else:
        inceptionV3 = models.inception_v3(weights=None)
    
    inceptionV3.fc = nn.Linear(inceptionV3.fc.in_features, cfg.model.num_classes)
    for param in inceptionV3.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in inceptionV3.fc.parameters():
        param.requires_grad = True
    
    if cfg.model.weight_path:
        inceptionV3.load_state_dict(torch.load(cfg.model.weight_path,weights_only=True))

    inceptionv3_transforms = transforms.Compose([
        transforms.Resize((299,299)),                
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    inceptionV3.aux_logits = False
    return inceptionV3, inceptionv3_transforms, None




def build_model(cfg: Config) -> ClassicModel:
    
    if cfg.model.name == "vgg16":
        return build_vgg16(cfg)
    elif cfg.model.name == "classic":
        return ClassicModel(cfg.model.in_channels, cfg.model.base_dim, cfg.model.num_classes)
    elif cfg.model.name == "resnet50":
        return build_resnet50(cfg)
    elif cfg.model.name == "inceptionV3":
        return build_inceptionV3(cfg)
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")
    
