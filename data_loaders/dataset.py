import csv
from pathlib import Path
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
from .transforms import image_transform

class ManifestDataset(Dataset):
    """CSV path,label[,group]. Every sample must have exactly one class label."""
    def __init__(self,root,manifest,image_size=84,train=False):
        self.root,self.manifest = Path(root).resolve(),Path(manifest).resolve()
        self.transform = image_transform(image_size,train)
        with self.manifest.open(encoding='utf-8-sig',newline='') as f:
            reader = csv.DictReader(f)
            if not {'path','label'} <= set(reader.fieldnames or []):
                raise ValueError(f'{manifest}: required CSV columns: path,label[,group]')
            self.records = list(reader)
        if not self.records:
            raise ValueError(f'Empty manifest: {manifest}')
        self.classes = sorted({r['label'] for r in self.records})
        mapping = {label:i for i,label in enumerate(self.classes)}
        self.targets = [mapping[r['label']] for r in self.records]
        self.paths = [(self.root/r['path']).resolve() for r in self.records]
        if len(set(self.paths))!=len(self.paths):
            raise ValueError(f'Duplicate paths in {manifest}; multilabel rows are unsupported')
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.groups = {r.get('group','').strip() for r in self.records}-{''}

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        with Image.open(self.paths[index]) as im:
            x = self.transform(im.convert('RGB'))
        return x,self.targets[index]

    def summary(self):
        return {'samples':len(self),'classes':len(self.classes),
                'counts':dict(Counter(r['label'] for r in self.records))}


def audit_splits(splits,policy='class_disjoint'):
    if policy not in ('class_disjoint','image_disjoint'):
        raise ValueError('split_policy must be class_disjoint or image_disjoint')
    names = list(splits)
    for i,name in enumerate(names):
        for other in names[i+1:]:
            a,b = splits[name],splits[other]
            if set(a.paths)&set(b.paths):
                raise ValueError(f'Image overlap between {name} and {other}')
            if a.groups&b.groups:
                raise ValueError(f'Group/patient overlap between {name} and {other}')
            if policy=='class_disjoint' and set(a.classes)&set(b.classes):
                raise ValueError(f'Class overlap between {name} and {other}')
    return {name:dataset.summary() for name,dataset in splits.items()}
