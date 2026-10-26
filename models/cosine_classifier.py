from torch import nn
from torch.nn import functional as F

class CosineClassifier(nn.Module):
    def __init__(self,num_classes,d=512,temperature=10.):
        super().__init__()
        self.fc = nn.Linear(d,num_classes,bias=False)
        self.temperature = temperature

    def forward(self,x):
        return self.temperature*F.linear(F.normalize(x,dim=-1),F.normalize(self.fc.weight,dim=-1))
