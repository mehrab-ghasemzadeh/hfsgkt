import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch
from data_loaders.dataset import ManifestDataset, audit_splits
from data_loaders.episode import EpisodeSampler
from utils.config import load_config
from utils.checkpoint import save_checkpoint, load_checkpoint

class DataTests(unittest.TestCase):
    def setUp(self):
        Path('work').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir='work')
        self.root = Path(self.tmp.name)
        rows = []
        for c in range(5):
            for i in range(5):
                path = self.root/f'{c}_{i}.png'
                Image.new('RGB',(16,16),(c*30,i*20,10)).save(path)
                rows.append({'path':path.name,'label':str(c),'group':f'{c}_{i}'})
        self.manifest = self.root/'train.csv'
        with self.manifest.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['path','label','group'])
            writer.writeheader(); writer.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_episodes_are_reproducible_disjoint_and_remap_labels(self):
        dataset = ManifestDataset(self.root,self.manifest,image_size=16)
        sampler = EpisodeSampler(dataset,5,1,2,seed=7)
        first = sampler.sample_indices(23)
        self.assertEqual(first,sampler.sample_indices(23))
        self.assertFalse(set(first['support']) & set(first['query']))
        self.assertEqual(set(first['support_y']),set(range(5)))
        sx,sy,qx,qy = sampler.sample(23)
        self.assertEqual(sx.shape,(5,3,16,16))
        self.assertEqual(qx.shape,(10,3,16,16))

    def test_insufficient_samples_fail_early(self):
        dataset = ManifestDataset(self.root,self.manifest,image_size=16)
        with self.assertRaisesRegex(ValueError,'eligible'):
            EpisodeSampler(dataset,5,5,15)

    def test_cross_split_leakage_is_rejected(self):
        dataset = ManifestDataset(self.root,self.manifest,image_size=16)
        with self.assertRaisesRegex(ValueError,'overlap'):
            audit_splits({'train':dataset,'test':dataset},'image_disjoint')

    def test_checkpoint_and_config_roundtrip(self):
        path = self.root/'state.pt'
        save_checkpoint({'stage':'test','step':4,'tensor':torch.ones(2)},path)
        self.assertEqual(load_checkpoint(path)['step'],4)
        (self.root/'base.yaml').write_text('model:\n  d: 16\nseed: 7\n')
        (self.root/'run.yaml').write_text('extends: base.yaml\nmodel:\n  heads: 4\n')
        cfg = load_config(self.root/'run.yaml',['model.d=32'])
        self.assertEqual(cfg['model'],{'d':32,'heads':4})
