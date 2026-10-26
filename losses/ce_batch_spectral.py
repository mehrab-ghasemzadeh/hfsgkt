import torch
from torch import nn
from torch.nn import functional as F

class BatchSpectralCE(nn.Module):
    def __init__(self,xi=.001,k=5):
        super().__init__()
        self.xi,self.k = xi,k

    def forward(self,logits,features,targets):
        ce = F.cross_entropy(logits,targets)
        if not self.xi:
            return ce
        singular = torch.linalg.svdvals(features.float())
        return ce+self.xi*singular[:self.k].sum()
