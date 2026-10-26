from torch import nn
from torch.nn import functional as F

class KLDomainLoss(nn.Module):
    """KL(uniform || prediction), as implemented by kl_div(log_p, uniform)."""
    def forward(self,source,target):
        return .5*sum(F.kl_div(x.log_softmax(-1),x.new_full(x.shape,.5),reduction='batchmean') for x in (source,target))
