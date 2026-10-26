from torch import nn
import torch
from torch.nn import functional as F

class PairwiseTransduction(nn.Module):
    """Query-conditioned MLP control; NOT the original KT free-coefficient solver."""
    def __init__(self,d=512,hidden_dim=512):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(3*d,hidden_dim),nn.LeakyReLU(.2),nn.Linear(hidden_dim,1))
        self.embedding = nn.Linear(d,d)

    def forward(self,queries,support):
        q = queries[:,None].expand(-1,len(support),-1)
        s = support[None].expand(len(queries),-1,-1)
        tau = self.score(torch.cat([q,s,q*s],-1)).squeeze(-1).softmax(-1)
        return tau,self.embedding(queries),self.embedding(s)
