import torch
import torch.nn as nn
import torch.nn.functional as F

from mobilenetv2 import MobileNetV2
from preresnet import PreResNet
from resnet import ResNet
from wideresnet import WideResNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

architectures = {
    'preresnet': PreResNet,
    'resnet': ResNet,
    'wideresnet': WideResNet,
    'mobilenetv2': MobileNetV2
}

def get_model(arch, num_classes, depth, widen_factor, simam=False):
    model = architectures[arch](num_classes=num_classes, depth=depth, widen_factor=widen_factor, attention=simam)
    return model.to(device)
