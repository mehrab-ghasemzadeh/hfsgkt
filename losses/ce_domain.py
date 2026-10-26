import torch
from torch import nn
from torch.nn import functional as F

class CEDomainLoss(nn.Module):
    def forward(self,source,target):
        return .5*(F.cross_entropy(source,torch.zeros(source.size(0),dtype=torch.long,device=source.device))+
                   F.cross_entropy(target,torch.ones(target.size(0),dtype=torch.long,device=target.device)))
