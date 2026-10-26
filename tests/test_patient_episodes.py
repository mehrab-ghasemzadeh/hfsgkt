import unittest
from types import SimpleNamespace
from data_loaders.episode import EpisodeSampler

class PatientEpisodeTests(unittest.TestCase):
    def test_support_and_query_have_disjoint_patient_groups(self):
        ds = SimpleNamespace(targets=[0]*12+[1]*12,classes=['a','b'])
        ds.records = [dict(group=str(i//3),path=str(i)) for i in range(24)]
        sampler = EpisodeSampler(ds,2,2,3,seed=42)
        for episode in range(15):
            record = sampler.sample_indices(episode)
            support_groups = {ds.records[i]['group'] for i in record['support']}
            query_groups = {ds.records[i]['group'] for i in record['query']}
            self.assertFalse(support_groups&query_groups)
