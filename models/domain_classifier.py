from torch import nn

class DomainClassifier(nn.Module):
    def __init__(self,feat_dim):
        super().__init__()
        self.fc = nn.Linear(feat_dim,2)

    def forward(self,x):
        return self.fc(x)
