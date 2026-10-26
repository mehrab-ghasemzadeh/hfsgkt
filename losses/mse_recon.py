from torch import nn

class MSEReconLoss(nn.Module):
    def forward(self,target,prediction):
        return (target-prediction).square().sum(-1).mean()
