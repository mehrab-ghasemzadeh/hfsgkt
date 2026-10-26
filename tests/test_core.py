import unittest
import torch
from models.hybrid_embed import HybridEmbedding
from models.gat_transduction import GATTransduction, build_cosine_topk_adjacency
from models.full_pipeline import EpisodicHead

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_hybrid_dynamic_weights(self):
        model = HybridEmbedding(d=16)
        mid, high, weights = model(torch.randn(7,128,8,8), torch.randn(7,256,4,4), torch.randn(7,512,2,2))
        self.assertEqual(mid.shape, (7,16))
        torch.testing.assert_close(mid.norm(dim=-1), torch.ones(7))
        torch.testing.assert_close(high.norm(dim=-1), torch.ones(7))
        torch.testing.assert_close(weights.sum(-1), torch.ones(7))
        mid.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.beta.parameters()))

    def test_one_shot_graph(self):
        graph = build_cosine_topk_adjacency(torch.randn(2,6,16),8)
        self.assertEqual(graph.shape,(2,6,6))
        self.assertTrue(graph.diagonal(dim1=-2,dim2=-1).all())
        self.assertTrue(torch.equal(graph,graph.transpose(-1,-2)))

    def test_support_permutation_and_query_independence(self):
        gat = GATTransduction(16,16,4,3,8,0).eval()
        support,queries = torch.randn(10,16),torch.randn(3,16)
        tau,_,_ = gat(queries,support)
        perm = torch.randperm(10)
        shuffled,_,_ = gat(queries,support[perm])
        torch.testing.assert_close(shuffled,tau[:,perm],atol=2e-6,rtol=2e-5)
        single,_,_ = gat(queries[:1],support)
        torch.testing.assert_close(single,tau[:1],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(tau.sum(-1),torch.ones(3))

    def test_query_predictions_and_gradient_for_both_branches(self):
        head = EpisodicHead(d=16,hidden_dim=16,num_heads=4,num_layers=3,k_neighbors=8,dropout=0,alpha=.5)
        out = head(torch.randn(5,16),torch.randn(10,16),torch.randn(5,16),torch.randn(10,16),torch.arange(5),5)
        self.assertEqual(out['scores'].shape,(10,5))
        torch.nn.functional.nll_loss(out['scores'],torch.arange(5).repeat(2)).backward()
        for name in ['adaptation','transduction','gnn_head']:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(head,name).parameters()),name)
        self.assertTrue(torch.isfinite(out['scores']).all())
