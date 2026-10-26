import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from data_loaders.episode import EpisodeSampler
from models.full_pipeline import Encoder,FullPipeline
from train.stage2 import head_from_config
from train.common import setup,datasets,verify_metadata
from utils.config import load_config
from utils.checkpoint import load_checkpoint


def summarize(accuracies):
    values = np.asarray(accuracies,dtype=np.float64)
    if len(values)<2:
        raise ValueError('At least two episodes required for a confidence interval')
    return dict(mean_accuracy_percent=float(100*values.mean()),
                ci95_percent=float(1.96*100*values.std(ddof=1)/math.sqrt(len(values))),episodes=len(values))


@torch.no_grad()
def evaluate(cfg,checkpoint,dataset_name,shot=1,episodes=None,output=None,overwrite=False,baseline=None):
    device = setup(cfg)
    state = load_checkpoint(checkpoint)
    verify_metadata(state,cfg)
    if baseline is None:
        if state['stage']!='stage2' or state['dataset']!=dataset_name or state['shot']!=shot:
            raise ValueError('Evaluation requires stage2 checkpoint matching dataset and shot')
    elif baseline!='prototype':
        raise ValueError('Unknown baseline')
    if baseline is None and state['config']['model']!=cfg['model']:
        raise ValueError('Checkpoint model differs from config')
    spec = cfg['datasets'][dataset_name]
    n_way = spec.get('n_way',5)
    data = datasets(cfg,dataset_name)
    test = data['test']
    settings = cfg['evaluation']
    count = episodes or settings['episodes']
    if count<2:
        raise ValueError('Evaluation requires at least 2 episodes')
    sampler = EpisodeSampler(test,n_way,shot,settings['n_query'],settings['seed'])
    encoder = Encoder(cfg['model']['d']).to(device)
    encoder.load_state_dict(state['models']['encoder'])
    head = head_from_config(cfg).to(device)
    if baseline is None:
        head.load_state_dict(state['models']['head'])
    pipeline = FullPipeline(encoder,head,settings['query_batch_size'],settings['encode_batch_size']).eval()
    directory = Path(output or f"{settings['output']}/{dataset_name}/{shot}shot/{baseline or 'hybrid'}")
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists() and not overwrite:
        raise FileExistsError(f'Results already exist: {directory}; choose --output or --overwrite')
    accuracies = []
    with (directory/'episodes.csv').open('w',newline='',encoding='utf-8') as table, (directory/'episode_indices.jsonl').open('w',encoding='utf-8') as records:
        writer = csv.DictWriter(table,fieldnames=['episode','accuracy','residual_accuracy','gnn_accuracy'])
        writer.writeheader()
        for episode in range(count):
            sx,sy,qx,qy = (x.to(device) for x in sampler.sample(episode))
            sm,sh = pipeline.encode(sx)
            qm,qh = pipeline.encode(qx)
            if baseline=='prototype':
                prototypes = torch.stack([sh[sy==c].mean(0) for c in range(n_way)])
                scores = -torch.cdist(qh,prototypes).square()
                residual = gnn = scores
            else:
                parts = [head(sm,m,sh,h,sy,n_way) for m,h in zip(qm.split(settings['query_batch_size']),qh.split(settings['query_batch_size']))]
                scores,residual,gnn = [torch.cat([part[key] for part in parts]) for key in ('scores','residual','gnn')]
            acc = (scores.argmax(-1)==qy).float().mean().item()
            accuracies.append(acc)
            writer.writerow(dict(episode=episode,accuracy=acc,residual_accuracy=(residual.argmax(-1)==qy).float().mean().item(),
                                 gnn_accuracy=(gnn.argmax(-1)==qy).float().mean().item()))
            ids = sampler.sample_indices(episode)
            ids['support_paths'] = [test.records[i]['path'] for i in ids['support']]
            ids['query_paths'] = [test.records[i]['path'] for i in ids['query']]
            records.write(json.dumps(dict(episode=episode,**ids))+'\n')
            if (episode+1)%100==0:
                print(f'{dataset_name} {shot}shot {episode+1}/{count}: {100*np.mean(accuracies):.2f}%',flush=True)
    result = dict(**summarize(accuracies),dataset=dataset_name,n_way=n_way,shot=shot,
                  n_query=settings['n_query'],seed=settings['seed'],checkpoint=str(checkpoint),
                  checkpoint_stage=state['stage'],method=baseline or 'hybrid',split_policy=spec.get('split_policy','class_disjoint'),
                  training_domain=state.get('training_domain'),warmup_targets=state.get('warmup_targets',[]),
                  training_seed=state['metadata']['seed'],metadata=state['metadata'],config=cfg,
                  torch_version=str(torch.__version__),device=str(device))
    (directory/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"{dataset_name} {n_way}-way {shot}-shot: {result['mean_accuracy_percent']:.2f} +/- {result['ci95_percent']:.2f}")
    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fixed-seed episodic evaluation without query-label adaptation')
    parser.add_argument('--config',default='configs/eval.yaml')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--shot',type=int,default=1)
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--output')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--baseline',choices=['prototype'])
    parser.add_argument('--set',action='append',default=[])
    args = parser.parse_args()
    evaluate(load_config(args.config,args.set),args.checkpoint,args.dataset,args.shot,args.episodes,args.output,args.overwrite,args.baseline)
