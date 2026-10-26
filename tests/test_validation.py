import unittest
import torch
from eval.validation import validate_pipeline

class ValidationTests(unittest.TestCase):
    def test_validation_restores_mode_and_does_not_update_batchnorm(self):
        class Sampler:
            n_way = 2
            def sample(self,index):
                x,y = torch.eye(2),torch.arange(2)
                return x,y,x,y
        class Pipeline(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2)
            def forward(self,support,queries,labels,n_way):
                return self.bn(queries)
        pipeline = Pipeline().train()
        before = pipeline.bn.running_mean.clone()
        score = validate_pipeline(pipeline,Sampler(),torch.device('cpu'),3)
        self.assertEqual(score,1.)
        self.assertTrue(pipeline.training)
        torch.testing.assert_close(before,pipeline.bn.running_mean)
