import unittest
import torch
from models.lagrange import LagrangeWeighting
from losses.kl_domain import KLDomainLoss
from losses.consistency import ConsistencyLoss
from models.full_pipeline import residual_logits
from eval.evaluate import summarize
from scripts.prepare_data import split_records

class ResearchTests(unittest.TestCase):
    def test_lagrange_four_terms_and_log_regularizer(self):
        weighting = LagrangeWeighting()
        total,phi = weighting([torch.tensor(float(i)) for i in range(1,5)])
        torch.testing.assert_close(phi,torch.full((4,),.25))
        torch.testing.assert_close(total,torch.tensor(2.5)-2*torch.tensor(.25).log())
        total.backward()
        self.assertGreater(weighting.raw.grad.abs().sum(),0)

    def test_uniform_kl_zero_and_consistency_stop_gradient(self):
        loss = KLDomainLoss()(torch.zeros(3,2),torch.zeros(4,2))
        torch.testing.assert_close(loss,torch.tensor(0.))
        x,y = torch.randn(3,8,requires_grad=True),torch.randn(3,8,requires_grad=True)
        ConsistencyLoss()(x,y).backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(y.grad)

    def test_residual_decision_recovers_known_reconstruction(self):
        support = torch.eye(3)
        tau = torch.tensor([[0.,1.,0.]])
        logits = residual_logits(support[1:2],tau,support,torch.arange(3),3)
        self.assertEqual(logits.argmax(-1).item(),1)

    def test_confidence_interval_uses_episode_sample_variance(self):
        report = summarize([0.,1.])
        self.assertEqual(report['mean_accuracy_percent'],50.)
        self.assertAlmostEqual(report['ci95_percent'],98.)

    def test_patient_groups_do_not_cross_generated_splits(self):
        records = [dict(path=f'{i}.png',label=str(i%2),group=str(i//3)) for i in range(60)]
        result = split_records(records,'images',42)
        groups = {name:{r['group'] for r in rows} for name,rows in result.items()}
        self.assertFalse(groups['train']&groups['test'])
        self.assertFalse(groups['train']&groups['val'])

    def test_class_split_filters_patients_with_multiple_labels(self):
        records = [dict(path=f'{c}_{i}.png',label=str(c),group=str(i)) for c in range(4) for i in range(6)]
        result = split_records(records,'classes',42,2,2)
        self.assertEqual(sum(map(len,result.values())),0)
