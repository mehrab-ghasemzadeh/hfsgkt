import torch
from torch import nn
from torch.nn import functional as F
from .backbone import ResNet18Backbone
from .hybrid_embed import HybridEmbedding
from .feature_adaptation import FeatureAdaptation
from .gat_transduction import GATTransduction
from .gnn_classifier import GNNLogitHead
from .projector import Projector
from .pairwise_transduction import PairwiseTransduction

class Encoder(nn.Module):
    def __init__(self,d=512):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.hybrid = HybridEmbedding(d)

    def forward(self,x):
        return self.hybrid(*self.backbone(x))

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self


def residual_logits(queries,tau,support,labels,n_way,temperature=10.):
    assignment = F.one_hot(labels,n_way).to(support.dtype)
    reconstructed = torch.einsum('bs,sd,sc->bcd',tau,support,assignment)
    return -temperature*(queries[:,None]-reconstructed).square().sum(-1)

class EpisodicHead(nn.Module):
    def __init__(self,d=512,hidden_dim=512,num_heads=4,num_layers=3,k_neighbors=8,dropout=.1,alpha=.5,residual_temperature=10.,transduction_type="gat",feature_source="mid"):
        super().__init__()
        if not 0<=alpha<=1:
            raise ValueError('alpha must be in [0,1]')
        if feature_source not in ('mid','high') or transduction_type not in ('gat','pairwise'):
            raise ValueError('Invalid feature source or transduction type')
        self.feature_source = feature_source
        self.alpha,self.residual_temperature = alpha,residual_temperature
        self.adaptation = FeatureAdaptation(d)
        self.transduction = (GATTransduction(d,hidden_dim,num_heads,num_layers,k_neighbors,dropout) if transduction_type=="gat" else PairwiseTransduction(d,hidden_dim))
        self.projector = Projector(d)
        self.gnn_head = GNNLogitHead(d)

    def forward(self,support,queries,support_high,query_high,labels,n_way):
        adapted = self.adaptation(support)
        tau,q_graph,s_graph = self.transduction(queries,adapted)
        residual = residual_logits(queries,tau,adapted,labels,n_way,self.residual_temperature)
        gnn = self.gnn_head(q_graph,s_graph,query_high,support_high,labels,n_way)
        # A weighted probability vote avoids mixing incompatible logit scales.
        probabilities = self.alpha*residual.softmax(-1)+(1-self.alpha)*gnn.softmax(-1)
        return dict(scores=probabilities.clamp_min(1e-12).log(),residual=residual,gnn=gnn,
                    tau=tau,reconstructed=tau@adapted,adapted=adapted)

class FullPipeline(nn.Module):
    def __init__(self,encoder,head,query_batch_size=16,encode_batch_size=128):
        super().__init__()
        self.encoder,self.head = encoder.freeze(),head
        self.query_batch_size,self.encode_batch_size = query_batch_size,encode_batch_size

    def train(self,mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self,images):
        mids,highs = [],[]
        for chunk in images.split(self.encode_batch_size):
            m,h,_ = self.encoder(chunk)
            mids.append(h if self.head.feature_source=="high" else m); highs.append(h)
        return torch.cat(mids),torch.cat(highs)

    def forward(self,support,queries,labels,n_way):
        sm,sh = self.encode(support)
        qm,qh = self.encode(queries)
        results = [self.head(sm,m,sh,h,labels,n_way)['scores'] for m,h in
                   zip(qm.split(self.query_batch_size),qh.split(self.query_batch_size))]
        return torch.cat(results)
