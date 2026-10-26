import argparse
import csv
import json
from pathlib import Path
from train.common import datasets
from data_loaders.episode import EpisodeSampler
from utils.config import load_config


def check(cfg,decode=False):
    report = {}
    for name in [cfg['source']]+cfg['targets']:
        splits = datasets(cfg,name)
        report[name] = {key:ds.summary() for key,ds in splits.items()}
        spec = cfg['datasets'][name]
        if name in cfg['targets']:
            EpisodeSampler(splits['test'],spec.get('n_way',5),max(cfg['evaluation']['shots']),cfg['evaluation']['n_query'])
            training_name = spec.get('train_domain',name)
            train = datasets(cfg,training_name)['train']
            EpisodeSampler(train,spec.get('train_n_way',spec.get('n_way',5)),max(cfg['evaluation']['shots']),cfg['stage2']['n_query'])
        if decode:
            for ds in splits.values():
                for index in range(len(ds)):
                    ds[index]
    return report

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/base.yaml')
    parser.add_argument('--decode-all',action='store_true')
    parser.add_argument('--output',default='results/data_audit.json')
    parser.add_argument('--set',action='append',default=[])
    args = parser.parse_args()
    report = check(load_config(args.config,args.set),args.decode_all)
    path = Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
