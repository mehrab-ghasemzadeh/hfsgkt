import torch
from torch import nn
from torchvision.models import resnet18

class ResNet18Backbone(nn.Module):
    """Standard torchvision stem; at 84px maps are 11/6/3, not 21/11/4."""
    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        self.stem = nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1)
        self.layer2,self.layer3,self.layer4 = net.layer2,net.layer3,net.layer4

    def forward(self,x):
        m2 = self.layer2(self.stem(x))
        m3 = self.layer3(m2)
        return m2,m3,self.layer4(m3)
