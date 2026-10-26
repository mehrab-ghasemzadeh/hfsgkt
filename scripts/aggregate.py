import argparse
import json
from pathlib import Path
import csv

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='results')
    parser.add_argument('--output',default='results/summary.csv')
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob('summary.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        rows.append({k:result.get(k) for k in ['dataset','method','n_way','shot','mean_accuracy_percent','ci95_percent','episodes','seed','split_policy','training_domain']})
    if not rows:
        raise SystemExit('No summary.json files found')
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(output)
