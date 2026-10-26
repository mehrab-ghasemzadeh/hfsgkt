"""Convert existing image datasets to portable manifests; never downloads data.

Use user-supplied split CSVs for exact published protocols. Automatic splits
are custom reproducible experiments and are not named official BSCD-FSL splits.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
CHEST_LABELS = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumothorax']


def folders(root):
    records = []
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and not directory.name.startswith('.'):
            for path in sorted(directory.rglob('*')):
                if path.suffix.lower() in EXTENSIONS and path.is_file():
                    records.append(dict(path=path.relative_to(root).as_posix(),label=directory.name,group=''))
    return records


def cub(root):
    # root must be extracted CUB_200_2011, not its outer archive directory.
    images = dict(line.strip().split(maxsplit=1) for line in (root/'images.txt').read_text().splitlines())
    labels = dict(line.strip().split(maxsplit=1) for line in (root/'image_class_labels.txt').read_text().splitlines())
    return [dict(path='images/'+images[key],label=labels[key],group='') for key in images]


def chest(root,metadata,labels):
    images = {}
    for path in sorted(root.rglob('*.png')):
        if path.name in images:
            raise ValueError(f'Duplicate ChestX filename: {path.name}')
        images[path.name] = path.relative_to(root).as_posix()
    records = []
    with metadata.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            pathology = row['Finding Labels']
            if '|' in pathology or pathology not in labels:
                continue  # excludes multilabel and No Finding; no arbitrary label selection
            filename = row['Image Index']
            if filename not in images:
                raise FileNotFoundError(f'ChestX image referenced in metadata missing: {filename}')
            records.append(dict(path=images[filename],label=pathology,group=row.get('Patient ID','').strip()))
    return records


def mini_csv(root,metadata):
    paths = {}
    for path in root.rglob('*'):
        if path.suffix.lower() in EXTENSIONS:
            if path.name in paths:
                raise ValueError(f'Ambiguous miniImageNet filename: {path.name}')
            paths[path.name] = path.relative_to(root).as_posix()
    records = []
    with metadata.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            filename = row.get('filename',row.get('path'))
            label = row.get('label',row.get('class'))
            if filename is None or label is None:
                raise ValueError('mini CSV needs filename,label (or path,class)')
            path = paths.get(Path(filename).name)
            if path is None:
                raise FileNotFoundError(filename)
            records.append(dict(path=path,label=label,group=''))
    return records


def split_records(records,mode,seed,train_classes=None,test_classes=None,train_fraction=.7,val_fraction=.1):
    rng = random.Random(seed)
    result = dict(train=[],val=[],test=[])
    classes = sorted({r['label'] for r in records})
    if mode=='none':
        return dict(test=records)
    if mode=='classes':
        rng.shuffle(classes)
        if train_classes is None:
            train_classes = len(classes)//2
        if test_classes is None:
            test_classes = len(classes)-train_classes
        if min(train_classes,test_classes)<1 or train_classes+test_classes>len(classes):
            raise ValueError('Invalid number of train/test classes')
        train_set = set(classes[:train_classes])
        test_set = set(classes[train_classes:train_classes+test_classes])
        for record in records:
            split = 'train' if record['label'] in train_set else 'test' if record['label'] in test_set else 'val'
            result[split].append(record)
        # A patient may occur under several pathology labels; remove cross-split patients.
        memberships = defaultdict(set)
        for split,rows in result.items():
            for row in rows:
                if row['group']:
                    memberships[row['group']].add(split)
        contaminated = {g for g,splits in memberships.items() if len(splits)>1}
        for split in result:
            result[split] = [r for r in result[split] if r['group'] not in contaminated]
    elif mode=='images':
        if not 0<train_fraction<1 or not 0<=val_fraction<1-train_fraction:
            raise ValueError('Fractions require 0 < train < 1 and 0 <= val < 1-train')
        # Global group assignment keeps each patient/lesion entirely within one split.
        groups = defaultdict(list)
        for i,record in enumerate(records):
            groups[record['group'] or f'__image_{i}'].append(record)
        keys = sorted(groups); rng.shuffle(keys)
        a,b = int(len(keys)*train_fraction),int(len(keys)*(train_fraction+val_fraction))
        for i,key in enumerate(keys):
            result['train' if i<a else 'val' if i<b else 'test'].extend(groups[key])
    else:
        raise ValueError(mode)
    return {name:rows for name,rows in result.items() if rows}


def write_manifests(splits,output,overwrite=False):
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in splits.items():
        path = output/f'{name}.csv'
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path}; use --overwrite to replace')
    for name,rows in splits.items():
        with (output/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['path','label','group'])
            writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format',required=True,choices=['folders','cub','chestx','mini_csv'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['classes','images','none'],default='classes')
    parser.add_argument('--train-classes',type=int)
    parser.add_argument('--test-classes',type=int)
    parser.add_argument('--train-fraction',type=float,default=.7)
    parser.add_argument('--val-fraction',type=float,default=.1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-name',choices=['train','val','test'],help='Use with --split none to preserve an existing split')
    parser.add_argument('--chest-labels',nargs='+',default=CHEST_LABELS)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.format=='folders':
        rows = folders(root)
    elif args.format=='cub':
        rows = cub(root)
    else:
        if args.metadata is None:
            parser.error('--metadata required for chestx/mini_csv')
        rows = chest(root,args.metadata,args.chest_labels) if args.format=='chestx' else mini_csv(root,args.metadata)
    if not rows:
        raise ValueError('No images found; --root must point directly to the dataset/class folders')
    missing = [row['path'] for row in rows if not (root/row['path']).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    splits = split_records(rows,args.split,args.seed,args.train_classes,args.test_classes,args.train_fraction,args.val_fraction)
    if args.output_name:
        if args.split!='none':
            parser.error('--output-name requires --split none')
        splits = {args.output_name:rows}
    write_manifests(splits,args.out,args.overwrite)
    report = dict(format=args.format,root=str(root),seed=args.seed,split=args.split,
                  counts={name:dict(samples=len(rs),classes=len({r['label'] for r in rs})) for name,rs in splits.items()},
                  warning='Automatically generated custom split; not verified as a published benchmark split')
    (args.out/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
