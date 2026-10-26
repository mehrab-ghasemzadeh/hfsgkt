from torch import nn
from torch.nn import functional as F

class FeatureAdaptation(nn.Module):
    def __init__(self,d=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d,d//2),nn.LeakyReLU(.1),nn.Linear(d//2,d),nn.Sigmoid())

    def forward(self,x):
        if x.ndim==4:
            x = x.mean((-2,-1))
        return x*self.gate(x)
