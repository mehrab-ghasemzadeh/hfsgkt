import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loaders.dataset import ManifestDataset,audit_splits
from utils.config import load_config
from utils.seed import set_seed,rng_state,restore_rng
from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.logger import TBLogger


def arguments(stage):
    parser = argparse.ArgumentParser(description=f'KT-GNN-Hybrid {stage}')
    parser.add_argument('--config',default=f'configs/{stage}.yaml')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    parser.add_argument('--resume',default=None)
    if stage=='stage1b':
        parser.add_argument('--init',required=False,default='checkpoints/stage1a/last.pt')
    if stage=='stage2':
        parser.add_argument('--init',default='checkpoints/stage1b/last.pt')
        parser.add_argument('--dataset',required=True)
        parser.add_argument('--shot',type=int,choices=[1,5,20,50],default=1)
    args = parser.parse_args()
    return args,load_config(args.config,args.set)


def setup(cfg):
    set_seed(cfg['seed'],cfg.get('deterministic',True))
    torch.set_num_threads(cfg.get('cpu_threads',4))
    name = cfg.get('device','auto')
    if name=='auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Check NVIDIA driver and PyTorch build.')
    return torch.device(name)


def datasets(cfg,name,augment_train=False):
    spec = cfg['datasets'][name]
    loaded = {split:ManifestDataset(spec['root'],path,cfg['image_size'],train=augment_train and split=='train')
              for split,path in spec['splits'].items() if path}
    audit_splits(loaded,spec.get('split_policy','class_disjoint'))
    return loaded


def fingerprint(cfg):
    result = {}
    names = {cfg['source'],*cfg['targets']}
    names |= {cfg['datasets'][name].get('train_domain',name) for name in list(names)}
    for name in sorted(names):
        spec = cfg['datasets'][name]
        for split,path in spec['splits'].items():
            if path:
                result[f'{name}/{split}'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def run_metadata(cfg):
    return dict(encoder_dim=cfg['model']['d'],source=cfg['source'],seed=cfg['seed'],
                image_size=cfg['image_size'],data=fingerprint(cfg))


def verify_metadata(state,cfg):
    if state['metadata']!=run_metadata(cfg):
        raise ValueError('Checkpoint model/data/seed differs from config; use matching manifests and settings')


def seed_worker(worker_id):
    seed = torch.initial_seed()%2**32
    random.seed(seed); np.random.seed(seed)


def loader(dataset,cfg,batch_size,epoch,shuffle=True):
    generator = torch.Generator().manual_seed(cfg['seed']+epoch)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                      num_workers=cfg.get('num_workers',4),pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker,generator=generator,drop_last=False)


def sample_batch(dataset,size,seed,device):
    generator = torch.Generator().manual_seed(seed)
    indices = (torch.randperm(len(dataset),generator=generator)[:size] if len(dataset)>=size else
               torch.randint(len(dataset),(size,),generator=generator)).tolist()
    batch = [dataset[i] for i in indices]
    return torch.stack([x for x,y in batch]).to(device),torch.tensor([y for x,y in batch],device=device)


def optimize(loss,optimizer,parameters,clip=5.):
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite training loss: {loss.detach().item()}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters,clip,error_if_nonfinite=True)
    optimizer.step()


def snapshot(stage,step,modules,optimizer,scheduler,cfg,**extra):
    return dict(stage=stage,step=step,models={k:v.state_dict() for k,v in modules.items()},
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                rng=rng_state(),config=cfg,metadata=run_metadata(cfg),**extra)


def restore(path,stage,modules,optimizer,scheduler,cfg):
    state = load_checkpoint(path,stage)
    verify_metadata(state,cfg)
    for name,module in modules.items():
        module.load_state_dict(state['models'][name])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler:
        scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    return state


def save_run_config(cfg,directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')


def frozen(module):
    return module.requires_grad_(False).eval()
