import json
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self,directory):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        self.writer = SummaryWriter(str(directory))
        self.handle = (directory/'metrics.jsonl').open('a',encoding='utf-8')
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

    def log(self,step,**scalars):
        values = {key:float(value) for key,value in scalars.items()}
        for key,value in values.items():
            self.writer.add_scalar(key,value,step)
        self.writer.flush()
        self.handle.write(json.dumps(dict(step=step,**values))+'\n')
        self.handle.flush()
        logging.info('step=%d %s',step,' '.join(f'{k}={v:.5f}' for k,v in values.items()))

    def close(self):
        self.writer.close()
        self.handle.close()
