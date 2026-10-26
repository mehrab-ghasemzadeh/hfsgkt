"""Predict an arbitrary support/query episode with a trained head, without query labels."""
import argparse
import csv
from pathlib import Path
import torch
from PIL import Image
from data_loaders.dataset import ManifestDataset
from data_loaders.transforms import image_transform
from models.full_pipeline import Encoder,FullPipeline
from train.stage2 import head_from_config
from train.common import setup
from utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--support',required=True,help='CSV path,label[,group]')
    parser.add_argument('--queries',required=True,help='CSV with path column; labels not needed')
    parser.add_argument('--output',default='results/predictions.csv',type=Path)
    parser.add_argument('--device',default='auto')
    args = parser.parse_args()
    state = load_checkpoint(args.checkpoint,'stage2')
    cfg = state['config']; cfg['device'] = args.device
    device = setup(cfg)
    support = ManifestDataset(args.root,args.support,cfg['image_size'])
    encoder = Encoder(cfg['model']['d']).to(device)
    head = head_from_config(cfg).to(device)
    encoder.load_state_dict(state['models']['encoder'])
    head.load_state_dict(state['models']['head'])
    pipeline = FullPipeline(encoder,head).eval()
    with Path(args.queries).open(encoding='utf-8-sig',newline='') as f:
        query_records = list(csv.DictReader(f))
    if not query_records or 'path' not in query_records[0]:
        raise ValueError('Non-empty query CSV with path column required')
    transform = image_transform(cfg['image_size'])
    sx = torch.stack([support[i][0] for i in range(len(support))]).to(device)
    sy = torch.tensor(support.targets,device=device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(),args.output.open('w',newline='',encoding='utf-8') as output:
        writer = csv.DictWriter(output,fieldnames=['path','prediction','confidence'])
        writer.writeheader()
        sm,sh = pipeline.encode(sx)
        for start in range(0,len(query_records),16):
            records = query_records[start:start+16]
            images = []
            for record in records:
                with Image.open(args.root/record['path']) as image:
                    images.append(transform(image.convert('RGB')))
            qm,qh = pipeline.encode(torch.stack(images).to(device))
            probability = head(sm,qm,sh,qh,sy,len(support.classes))['scores'].exp()
            confidence,prediction = probability.max(-1)
            for record,label,score in zip(records,prediction.tolist(),confidence.tolist()):
                writer.writerow(dict(path=record['path'],prediction=support.classes[label],confidence=score))
    print(args.output)

if __name__=='__main__':
    main()
