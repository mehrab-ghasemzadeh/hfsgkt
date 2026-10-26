"""Offline end-to-end numerical smoke test on generated images, not real accuracy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from utils.config import load_config
from utils.checkpoint import load_checkpoint
from train.stage1a import train as stage1a
from train.stage1b import train as stage1b
from train.stage2 import train as stage2
from eval.evaluate import evaluate
from scripts.check_data import check


def create_images(root,n_classes,per_class,seed):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True,exist_ok=True)
    records = []
    for cls in range(n_classes):
        for index in range(per_class):
            name = f'class{cls}_{index}.png'
            pixels = rng.integers(0,180,(32,32,3),dtype=np.uint8)
            pixels[:,:,cls%3] = min(250,80+cls*15)
            Image.fromarray(pixels).save(root/name)
            records.append(dict(path=name,label=str(cls),group=''))
    return records


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=['path','label','group'])
        writer.writeheader(); writer.writerows(rows)


def run(directory,device='auto'):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    source_root,target_root = directory/'source',directory/'target'
    source = create_images(source_root,4,8,0)
    target = create_images(target_root,10,8,1)
    write_csv(directory/'source.csv',source)
    write_csv(directory/'target_train.csv',[r for r in target if int(r['label'])<5])
    write_csv(directory/'target_test.csv',[r for r in target if int(r['label'])>=5])
    cfg = load_config('configs/base.yaml')
    cfg.update(device=device,num_workers=0,cpu_threads=2,image_size=32,source='source',targets=['target'])
    cfg['model']['d'] = 32
    cfg['model']['head'].update(hidden_dim=32,num_heads=4,num_layers=2)
    cfg['datasets'] = {
        'source':dict(root=str(source_root),split_policy='class_disjoint',splits=dict(train=str(directory/'source.csv'))),
        'target':dict(root=str(target_root),split_policy='class_disjoint',n_way=5,train_n_way=5,
                      splits=dict(train=str(directory/'target_train.csv'),test=str(directory/'target_test.csv')))}
    for stage in ('stage1a','stage1b','stage2'):
        cfg[stage]['output'] = str(directory/stage)
        cfg[stage]['log_dir'] = str(directory/'runs'/stage)
        cfg[stage]['save_every'] = 1
    cfg['stage1a'].update(epochs=1,batch_size=8,max_batches=1)
    cfg['stage1b'].update(steps=2,source_batch_size=4,target_batch_size=4,log_every=1)
    cfg['stage2'].update(steps=2,n_query=1,query_batch_size=5,encode_batch_size=16,log_every=1)
    cfg['evaluation'].update(episodes=3,n_query=1,shots=[1,5],query_batch_size=5,encode_batch_size=16,output=str(directory/'results'))
    (directory/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    check(cfg,decode=True)
    first = stage1a(cfg)
    second = stage1b(cfg,first)
    before = load_checkpoint(second)['models']['encoder']
    third = stage2(cfg,second,'target',1)
    uninterrupted = load_checkpoint(third)
    # Frozen encoder includes BN running statistics, not just parameters.
    for name,tensor in before.items():
        torch.testing.assert_close(tensor,uninterrupted['models']['encoder'][name],rtol=0,atol=0)
    stage2(cfg,second,'target',1,stop_after=1)
    stage2(cfg,second,'target',1,resume=third)
    resumed = load_checkpoint(third)
    for name,tensor in uninterrupted['models']['head'].items():
        torch.testing.assert_close(tensor,resumed['models']['head'][name],rtol=0,atol=0)
    result1 = evaluate(cfg,third,'target',1,overwrite=True)
    fifth = stage2(cfg,second,'target',5)
    result5 = evaluate(cfg,fifth,'target',5,overwrite=True)
    report = dict(status='passed',device=device,cuda_available=torch.cuda.is_available(),torch=str(torch.__version__),
                  tests=['dataset_audit','stage1a_forward_backward','stage1b_domain_alignment','stage2_both_heads',
                         'frozen_encoder_and_batchnorm','bitwise_resume','1shot_eval','5shot_eval'],
                  synthetic_only=True,results=[result1,result5])
    (directory/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SMOKE PASSED: full pipeline, frozen BN, exact resume, 1-shot and 5-shot')
    return report

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',default='work/smoke')
    parser.add_argument('--device',default='auto')
    args = parser.parse_args()
    run(args.work_dir,args.device)
