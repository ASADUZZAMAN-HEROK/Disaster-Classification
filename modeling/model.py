import torch
import torch.nn as nn
from torchvision import models, transforms

from configs import Config
from typing import Tuple


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


##Ready
def build_vgg16(cfg: Config):   
    vgg16 = models.vgg16()
    
    if cfg.model.pretrained:
        vgg16 = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1, progress = False)
    else:
        vgg16 = models.vgg16(weights=None)
    
    vgg16.classifier[6] = nn.Linear(4096, cfg.model.num_classes)
    for param in vgg16.features.parameters():
        param.requires_grad = not cfg.model.pretrained
    for i in range(6):
        vgg16.classifier[i].requires_grad = True

    if cfg.model.weight_path:
        vgg16.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))
    
    vgg16_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return vgg16, vgg16_transforms, None

#Ready
def build_resnet50(cfg: Config):
    resnet50 = models.resnet50()
    
    if cfg.model.pretrained:
        resnet50 = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1, progress = False,)
    else:
        resnet50 = models.resnet50(weights=None)
    
    resnet50.fc = nn.Linear(2048, cfg.model.num_classes)
    for param in resnet50.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in resnet50.fc.parameters():
        param.requires_grad = True
    
    if cfg.model.weight_path:
        resnet50.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    resnet50_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return resnet50, resnet50_transforms, None

#Ready
def build_inceptionV3(cfg: Config):
    if cfg.model.pretrained:
        inceptionV3 = models.inception_v3(weights=models.inception.Inception_V3_Weights.IMAGENET1K_V1, progress = False)
    else:
        inceptionV3 = models.inception_v3(weights=None)
    
    inceptionV3.fc = nn.Linear(inceptionV3.fc.in_features, cfg.model.num_classes)
    for param in inceptionV3.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in inceptionV3.fc.parameters():
        param.requires_grad = True
    
    if cfg.model.weight_path:
        inceptionV3.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    inceptionv3_transforms = transforms.Compose([
        transforms.Resize((299, 299)),                
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    inceptionV3.aux_logits = False
    return inceptionV3, inceptionv3_transforms, None


#Ready
def build_densenet121(cfg: Config):
    densenet121 = models.densenet121()

    if cfg.model.pretrained:
        densenet121 = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1, progress=False)
    else:
        densenet121 = models.densenet121(weights=None)
    
    densenet121.classifier = nn.Linear(densenet121.classifier.in_features, cfg.model.num_classes)
    for param in densenet121.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in densenet121.classifier.parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        densenet121.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    densenet_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return densenet121, densenet_transforms, None

#Ready
def build_mobilenet_v2(cfg: Config):
    mobilenet_v2 = models.mobilenet_v2()

    if cfg.model.pretrained:
        mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, progress = False)
    else:
        mobilenet_v2 = models.mobilenet_v2(weights=None)
    
    mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.classifier[1].in_features, cfg.model.num_classes)
    for param in mobilenet_v2.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in mobilenet_v2.classifier[1].parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        mobilenet_v2.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    mobilenet_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return mobilenet_v2, mobilenet_transforms, None

#Ready
def build_efficientnet_b0(cfg: Config):
    efficientnet_b0 = models.efficientnet_b0()

    if cfg.model.pretrained:
        efficientnet_b0 = models.efficientnet_b0(weights=models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1,progress=False)
    else:
        efficientnet_b0 = models.efficientnet_b0(weights=None)
    
    efficientnet_b0.classifier[1] = nn.Linear(efficientnet_b0.classifier[1].in_features, cfg.model.num_classes)
    for param in efficientnet_b0.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in efficientnet_b0.classifier[1].parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        efficientnet_b0.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    efficientnet_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return efficientnet_b0, efficientnet_transforms, None


#Ready
def build_alexnet(cfg: Config):
    alexnet = models.alexnet()

    if cfg.model.pretrained:
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1, progress = False)
    else:
        alexnet = models.alexnet(weights=None)
    
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, cfg.model.num_classes)
    for param in alexnet.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in alexnet.classifier[6].parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        alexnet.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    alexnet_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return alexnet, alexnet_transforms, None

#Ready
def build_squeezenet1_0(cfg: Config):
    squeezenet = models.squeezenet1_0()

    if cfg.model.pretrained:
        squeezenet = models.squeezenet1_0(weights=models.squeezenet.SqueezeNet1_0_Weights.IMAGENET1K_V1, progress = False)
    else:
        squeezenet = models.squeezenet1_0(weights=None)
    
    squeezenet.classifier[1] = nn.Conv2d(512, cfg.model.num_classes, kernel_size=(1, 1))
    squeezenet.classifier[1].bias.requires_grad = True  # make sure biases are trainable

    if cfg.model.weight_path:
        squeezenet.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    squeezenet_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return squeezenet, squeezenet_transforms, None

#Ready
def build_shufflenet_v2_x1_0(cfg: Config):
    shufflenet_v2 = models.shufflenet_v2_x1_0()

    if cfg.model.pretrained:
        shufflenet_v2 = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1, progress = False)
    else:
        shufflenet_v2 = models.shufflenet_v2_x1_0(weights=None)

    shufflenet_v2.fc = nn.Linear(shufflenet_v2.fc.in_features, cfg.model.num_classes)
    for param in shufflenet_v2.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in shufflenet_v2.fc.parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        shufflenet_v2.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    shufflenet_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return shufflenet_v2, shufflenet_transforms, None

#Ready
def build_mnasnet1_0(cfg: Config):
    mnasnet = models.mnasnet1_0()

    if cfg.model.pretrained:
        mnasnet = models.mnasnet1_0(weights=models.mnasnet.MNASNet1_0_Weights.IMAGENET1K_V1, progress = False)
    else:
        mnasnet = models.mnasnet1_0(weights=None)

    mnasnet.classifier[1] = nn.Linear(mnasnet.classifier[1].in_features, cfg.model.num_classes)
    for param in mnasnet.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in mnasnet.classifier[1].parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        mnasnet.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    mnasnet_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return mnasnet, mnasnet_transforms, None

#Ready
def build_regnet_y_400mf(cfg: Config):
    regnet_y_400mf = models.regnet_y_400mf()

    if cfg.model.pretrained:
        regnet_y_400mf = models.regnet_y_400mf(weights=models.regnet.RegNet_Y_400MF_Weights.IMAGENET1K_V1, progress = False)
    else:
        regnet_y_400mf = models.regnet_y_400mf(weights=None)

    regnet_y_400mf.fc = nn.Linear(regnet_y_400mf.fc.in_features, cfg.model.num_classes)
    for param in regnet_y_400mf.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in regnet_y_400mf.fc.parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        regnet_y_400mf.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    regnet_y_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return regnet_y_400mf, regnet_y_transforms, None

#Ready
def build_vit_b_16(cfg: Config):
    vit_b_16 = models.vit_b_16()

    if cfg.model.pretrained:
        vit_b_16 = models.vit_b_16(weights=models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1,progress=False)
    else:
        vit_b_16 = models.vit_b_16(weights=None)

    vit_b_16.heads[0] = nn.Linear(vit_b_16.heads[0].in_features, cfg.model.num_classes)
    for param in vit_b_16.parameters():
        param.requires_grad = not cfg.model.pretrained
    for param in vit_b_16.heads[0].parameters():
        param.requires_grad = True

    if cfg.model.weight_path:
        vit_b_16.load_state_dict(torch.load(cfg.model.weight_path, weights_only=True))

    vit_b_16_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.CenterCrop(224),            
        transforms.ToTensor(),                 
        transforms.Normalize(                  
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return vit_b_16, vit_b_16_transforms, None


def build_model(cfg: Config) -> Tuple[nn.Module, transforms.Compose, None]:
    
    if cfg.model.name == "vgg16":
        return build_vgg16(cfg)
    elif cfg.model.name == "resnet50":
        return build_resnet50(cfg)
    elif cfg.model.name == "inceptionV3":
        return build_inceptionV3(cfg)
    elif cfg.model.name == "densenet121":
        return build_densenet121(cfg)
    elif cfg.model.name == "mobilenet_v2":
        return build_mobilenet_v2(cfg)
    elif cfg.model.name == "efficientnet_b0":
        return build_efficientnet_b0(cfg)
    elif cfg.model.name == "alexnet":
        return build_alexnet(cfg)
    elif cfg.model.name == "squeezenet1_0":
        return build_squeezenet1_0(cfg)
    elif cfg.model.name == "shufflenet_v2_x1_0":
        return build_shufflenet_v2_x1_0(cfg)
    elif cfg.model.name == "mnasnet1_0":
        return build_mnasnet1_0(cfg)
    elif cfg.model.name == "regnet_y_400mf":
        return build_regnet_y_400mf(cfg)
    elif cfg.model.name == "vit_b_16":
        return build_vit_b_16(cfg)
    elif cfg.model.name == "classic":
        return ClassicModel(cfg.model.in_channels, cfg.model.base_dim, cfg.model.num_classes)
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")

    
