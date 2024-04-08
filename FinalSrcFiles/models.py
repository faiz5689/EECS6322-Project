import torch
import torch.nn as nn
import torch.nn.functional as F

from mobilenetv2 import MobileNetV2
from preresnet import PreResNet
from resnet import ResNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

architectures = {
    'preresnet': PreResNet,
    'resnet': ResNet,
    'mobilenetv2': MobileNetV2
}

def get_model(arch, num_classes, depth, simam=False):
    model = architectures[arch](num_classes=num_classes, depth=depth, attention=simam)
    return model.to(device)
