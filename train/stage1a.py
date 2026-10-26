import torch
from torch import nn
from data_loaders.episode import EpisodeSampler
from eval.validation import validate_encoder
from models.full_pipeline import Encoder
from models.cosine_classifier import CosineClassifier
from losses.ce_batch_spectral import BatchSpectralCE
from utils.checkpoint import save_checkpoint
from utils.logger import TBLogger
from .common import arguments,setup,datasets,loader,optimize,snapshot,restore,save_run_config


def train(cfg,resume=None):
    device = setup(cfg)
    data = datasets(cfg,cfg['source'],augment_train=True)
    if 'train' not in data:
        raise ValueError('Source requires train manifest')
    encoder = Encoder(cfg['model']['d']).to(device)
    classifier = CosineClassifier(len(data['train'].classes),cfg['model']['d']).to(device)
    modules = dict(encoder=encoder,classifier=classifier)
    parameters = list(encoder.parameters())+list(classifier.parameters())
    settings = cfg['stage1a']
    optimizer = torch.optim.SGD(parameters,lr=settings['lr'],momentum=.9,weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,settings['epochs'],eta_min=settings['min_lr'])
    criterion = BatchSpectralCE(cfg['loss']['spectral_weight'],cfg['loss']['spectral_k'])
    validation = EpisodeSampler(data['val'],5,1,settings.get('validation_query',15),cfg['seed']+100000) if 'val' in data else None
    best = -1.
    start = 0
    if resume:
        state = restore(resume,'stage1a',modules,optimizer,scheduler,cfg)
        start = state['step']
        best = state.get('best_validation_accuracy',-1.)
        if state['config']['stage1a']!=settings:
            raise ValueError('Resume requires unchanged stage1a schedule')
    directory = settings['output']
    save_run_config(cfg,directory)
    logger = TBLogger(settings['log_dir'])
    try:
        for epoch in range(start,settings['epochs']):
            encoder.train(); classifier.train()
            total,correct,count = 0.,0,0
            for batch_index,(x,y) in enumerate(loader(data['train'],cfg,settings['batch_size'],epoch)):
                x,y = x.to(device),y.to(device)
                mid,high,_ = encoder(x)
                # Both paths receive classification gradients from pretraining onwards.
                logits_high = classifier(high)
                loss = .5*(criterion(logits_high,high,y)+criterion(classifier(mid),mid,y))
                optimize(loss,optimizer,parameters)
                total += loss.item()*y.numel()
                correct += int((logits_high.argmax(-1)==y).sum())
                count += y.numel()
                if settings.get('max_batches') and batch_index+1>=settings['max_batches']:
                    break
            scheduler.step()
            logger.log(epoch+1,loss=total/count,accuracy=correct/count,lr=optimizer.param_groups[0]['lr'])
            improved = False
            if validation and (epoch+1)%settings.get('validation_every',10)==0:
                score = validate_encoder(encoder,validation,device,settings.get('validation_episodes',50))
                logger.log(epoch+1,validation_accuracy=score)
                improved = score>best
                best = max(best,score)
            state = snapshot('stage1a',epoch+1,modules,optimizer,scheduler,cfg,source_classes=data['train'].classes,best_validation_accuracy=best)
            if improved:
                save_checkpoint(state,f'{directory}/best.pt')
            save_checkpoint(state,f'{directory}/last.pt')
            if (epoch+1)%settings['save_every']==0:
                save_checkpoint(state,f'{directory}/epoch_{epoch+1:04d}.pt')
    finally:
        logger.close()
    return f'{directory}/last.pt'

if __name__=='__main__':
    args,cfg = arguments('stage1a')
    train(cfg,args.resume)
