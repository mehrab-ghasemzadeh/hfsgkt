from torch import nn
from torch.nn import functional as F

class ConsistencyLoss(nn.Module):
    def forward(self,projection,target):
        return -F.cosine_similarity(projection,target.detach(),dim=-1).mean()
