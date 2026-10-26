from pathlib import Path
import os
import torch

def save_checkpoint(state,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)

def load_checkpoint(path,stage=None):
    state = torch.load(path,map_location='cpu',weights_only=True)
    if stage is not None and state.get('stage')!=stage:
        raise ValueError(f'Expected {stage} checkpoint, got {state.get("stage")}')
    return state
