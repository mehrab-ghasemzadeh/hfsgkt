import torch
from torch import nn
from torch.nn import functional as F

class HybridEmbedding(nn.Module):
    def __init__(self,d=512):
        super().__init__()
        self.beta = nn.ModuleList([nn.Linear(c,1) for c in (128,256)])
        self.transforms = nn.ModuleList([nn.Linear(c,d) for c in (128,256)])
        self.high = nn.Identity() if d==512 else nn.Linear(512,d)

    def forward(self,m2,m3,m4):
        pooled = [m.mean((-2,-1)) for m in (m2,m3)]
        weights = torch.cat([b(m) for b,m in zip(self.beta,pooled)],-1).softmax(-1)
        layers = torch.stack([F.normalize(w(m),dim=-1) for w,m in zip(self.transforms,pooled)],1)
        mid = F.normalize((layers*weights.unsqueeze(-1)).sum(1),dim=-1)
        high = F.normalize(self.high(m4.mean((-2,-1))),dim=-1)
        return mid,high,weights
