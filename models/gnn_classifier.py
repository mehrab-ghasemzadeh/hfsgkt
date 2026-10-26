import torch
from torch import nn
from torch.nn import functional as F

class GNNLogitHead(nn.Module):
    def __init__(self,d=512,temperature=10.):
        super().__init__()
        self.fuse = nn.Linear(2*d,d)
        self.temperature = temperature

    def forward(self,query_graph,support_graph,query_high,support_high,labels,n_way):
        q = F.normalize(self.fuse(torch.cat([query_graph,query_high],-1)),dim=-1)
        high = support_high[None].expand(support_graph.size(0),-1,-1)
        s = self.fuse(torch.cat([support_graph,high],-1))
        assignment = F.one_hot(labels,n_way).to(s.dtype)
        prototypes = torch.einsum('bsd,sc->bcd',s,assignment)/assignment.sum(0)[None,:,None]
        return self.temperature*torch.einsum('bd,bcd->bc',q,F.normalize(prototypes,dim=-1))
