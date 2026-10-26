"""Launch configurable training/evaluation from one command; subprocess failures stop the run."""
import argparse
import subprocess
import sys
from pathlib import Path
from utils.config import load_config


def run(command):
    print(' '.join(command),flush=True)
    subprocess.run(command,check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/source_only.yaml')
    parser.add_argument('--profile',choices=['source_only','target_warmup'],default='source_only')
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--shots',type=int,nargs='+',default=[1,5])
    parser.add_argument('--skip-stage1a',action='store_true')
    parser.add_argument('--stage1a-checkpoint')
    parser.add_argument('--set',action='append',default=[])
    args = parser.parse_args()
    cfg = load_config(args.config,args.set)
    names = args.datasets or cfg['targets']
    if args.profile=='source_only':
        invalid = [name for name in names if cfg['datasets'][name].get('train_domain',name)!=cfg['source']]
        if invalid:
            parser.error('source_only requires configs/source_only.yaml (or train_domain=source for every target)')
    common = ['--config',args.config]
    for override in args.set:
        common += ['--set',override]
    command = [sys.executable,'-m']
    run(command+['scripts.check_data']+common)
    if not args.skip_stage1a:
        run(command+['train.stage1a']+common)
    initial = args.stage1a_checkpoint or f"{cfg['stage1a']['output']}/last.pt"
    if not Path(initial).is_file():
        raise FileNotFoundError(initial)
    for name in names:
        init = initial
        if args.profile=='target_warmup' and cfg['datasets'][name].get('warmup',True):
            # Isolated target warmup: no shared training across held-out target domains.
            output = f"{cfg['stage1b']['output']}/{name}"
            run(command+['train.stage1b']+common+['--init',initial,
                '--set',f'stage1b.warmup_targets=[{name}]','--set',f'stage1b.output={output}',
                '--set',f"stage1b.log_dir={cfg['stage1b']['log_dir']}/{name}"])
            init = f'{output}/last.pt'
        for shot in args.shots:
            run(command+['train.stage2']+common+['--init',init,'--dataset',name,'--shot',str(shot)])
            checkpoint = f"{cfg['stage2']['output']}/{name}/{shot}shot/last.pt"
            run(command+['eval.evaluate']+common+['--checkpoint',checkpoint,'--dataset',name,'--shot',str(shot)])

if __name__=='__main__':
    main()
