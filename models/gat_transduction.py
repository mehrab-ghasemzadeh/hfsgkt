import torch
from torch import nn
from torch.nn import functional as F

@torch.no_grad()
def build_cosine_topk_adjacency(features,k=8):
    """Union-symmetrized kNN, adjacency[receiver,sender], with self loops."""
    if k<1 or features.size(-2)<2:
        raise ValueError('Graph needs at least two nodes and k >= 1')
    n = features.size(-2)
    normalized = F.normalize(features,dim=-1)
    similarity = normalized @ normalized.transpose(-1,-2)
    eye = torch.eye(n,dtype=torch.bool,device=features.device)
    similarity = similarity.masked_fill(eye,float('-inf'))
    indices = similarity.topk(min(k,n-1),dim=-1).indices
    adj = torch.zeros_like(similarity,dtype=torch.bool).scatter_(-1,indices,True)
    return adj | adj.transpose(-1,-2) | eye

class GATLayer(nn.Module):
    """Dense batched standard additive multi-head GAT for small episode graphs."""
    def __init__(self,in_dim,out_dim,heads,dropout):
        super().__init__()
        if out_dim%heads:
            raise ValueError('hidden_dim must be divisible by heads')
        self.heads,self.width,self.dropout = heads,out_dim//heads,dropout
        self.projection = nn.Linear(in_dim,out_dim,bias=False)
        self.left = nn.Parameter(torch.empty(heads,self.width))
        self.right = nn.Parameter(torch.empty(heads,self.width))
        nn.init.xavier_uniform_(self.left)
        nn.init.xavier_uniform_(self.right)
        self.residual = nn.Linear(in_dim,out_dim,bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self,h,adj):
        b,n,_ = h.shape
        z = self.projection(h).view(b,n,self.heads,self.width).transpose(1,2)
        left,right = (z*self.left[None,:,None,:]).sum(-1),(z*self.right[None,:,None,:]).sum(-1)
        e = F.leaky_relu(left.unsqueeze(-1)+right.unsqueeze(-2),.2)
        attention = e.masked_fill(~adj[:,None],float('-inf')).softmax(-1)
        value = F.dropout(attention,self.dropout,self.training) @ z
        value = value.transpose(1,2).reshape(b,n,-1)
        return F.elu(self.norm(value+self.residual(h))),attention

class GATTransduction(nn.Module):
    def __init__(self,feat_dim=512,hidden_dim=512,num_heads=4,num_layers=3,k_neighbors=8,dropout=.1):
        super().__init__()
        if num_layers<1:
            raise ValueError('num_layers must be positive')
        self.k_neighbors = k_neighbors
        self.layers = nn.ModuleList([GATLayer(feat_dim if i==0 else hidden_dim,hidden_dim,num_heads,dropout) for i in range(num_layers)])
        self.embedding = nn.Linear(hidden_dim,feat_dim)

    def forward(self,queries,support):
        # Each query gets its own graph: no averaging and no query-query leakage.
        b = queries.size(0)
        nodes = torch.cat([queries[:,None,:],support[None].expand(b,-1,-1)],1)
        adjacency = build_cosine_topk_adjacency(nodes,self.k_neighbors)
        h = nodes
        for layer in self.layers:
            h,attention = layer(h,adjacency)
        tau = attention.mean(1)[:,0,1:]
        tau = tau/tau.sum(-1,keepdim=True).clamp_min(1e-12)
        projected = self.embedding(h)
        return tau,projected[:,0],projected[:,1:]
