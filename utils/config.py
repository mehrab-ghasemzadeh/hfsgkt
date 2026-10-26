from pathlib import Path
import os
import yaml

def merge(base,update):
    result = dict(base)
    for key,value in update.items():
        result[key] = merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else value
    return result

def load_config(path,overrides=(),_seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f'Config inheritance cycle: {path}')
    seen.add(path)
    with path.open(encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(os.path.expandvars(f.read())) or {}
    parent = cfg.pop('extends',None)
    if parent:
        cfg = merge(load_config(path.parent/parent,_seen=seen),cfg)
    for override in overrides:
        if '=' not in override:
            raise ValueError(f'Override needs key=value: {override}')
        name,value = override.split('=',1)
        target = cfg
        keys = name.split('.')
        for key in keys[:-1]:
            target = target.setdefault(key,{})
        target[keys[-1]] = yaml.safe_load(value)
    return cfg
