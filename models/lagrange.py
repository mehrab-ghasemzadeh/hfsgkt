import torch
from torch import nn

class LagrangeWeighting(nn.Module):
    """HFDA Eq.10-11: four positive task weights, exact simplex constraint.

    The constraint is eliminated via softmax; lambda*(sum(phi)-1) is identically
    zero. We retain the paper's -0.5*sum(log(phi)), absent in the supplied guide.
    This is constrained optimization, not an implementation of dual ascent Eq.12.
    """
    def __init__(self,n_tasks=4):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(n_tasks))

    def forward(self,losses):
        if len(losses)!=self.raw.numel():
            raise ValueError('One task weight required per loss')
        log_phi = self.raw.log_softmax(0)
        phi = log_phi.exp()
        return (phi*torch.stack(losses)).sum()-.5*log_phi.sum(),phi.detach()
