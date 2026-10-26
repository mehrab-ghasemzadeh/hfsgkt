import torch
from torch.nn import functional as F
from models.full_pipeline import Encoder,EpisodicHead,FullPipeline
from eval.validation import validate_pipeline
from data_loaders.episode import EpisodeSampler
from data_loaders.transforms import StochasticPixelMask
from losses.mse_recon import MSEReconLoss
from losses.consistency import ConsistencyLoss
from utils.checkpoint import load_checkpoint,save_checkpoint
from utils.logger import TBLogger
from .common import arguments,setup,datasets,optimize,snapshot,restore,save_run_config,verify_metadata


def head_from_config(cfg):
    return EpisodicHead(d=cfg['model']['d'],**cfg['model']['head'])


def train(cfg,init,dataset_name,shot=1,resume=None,stop_after=None):
    device = setup(cfg)
    spec = cfg['datasets'][dataset_name]
    training_name = spec.get('train_domain',dataset_name)
    training_splits = datasets(cfg,training_name)
    ds = training_splits['train']
    # Training can have fewer classes than test: no output unit is tied to a class ID.
    n_way = spec.get('train_n_way',spec.get('n_way',5))
    settings = cfg['stage2']
    sampler = EpisodeSampler(ds,n_way,shot,settings['n_query'],cfg['seed'])
    encoder = Encoder(cfg['model']['d']).to(device).freeze()
    head = head_from_config(cfg).to(device)
    pipeline = FullPipeline(encoder,head,settings['query_batch_size'],settings['encode_batch_size'])
    initial = load_checkpoint(init)
    if initial['stage'] not in ('stage1a','stage1b'):
        raise ValueError('Stage2 --init needs stage1a or stage1b checkpoint')
    verify_metadata(initial,cfg)
    encoder.load_state_dict(initial['models']['encoder'])
    parameters = list(head.parameters())
    optimizer = torch.optim.Adam(parameters,lr=settings['lr'],weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,settings['step_size'],settings['gamma'])
    modules = dict(encoder=encoder,head=head)
    validation = (EpisodeSampler(training_splits['val'],n_way,shot,settings['n_query'],cfg['seed']+100000)
                  if 'val' in training_splits else None)
    best = -1.
    start = 0
    if resume:
        state = restore(resume,'stage2',modules,optimizer,scheduler,cfg)
        if state['dataset']!=dataset_name or state['shot']!=shot or state['config']['stage2']!=settings or state['config']['model']!=cfg['model']:
            raise ValueError('Resume dataset/shot/training settings mismatch')
        start = state['step']
        best = state.get('best_validation_accuracy',-1.)
    directory = f"{settings['output']}/{dataset_name}/{shot}shot"
    save_run_config(cfg,directory)
    logger = TBLogger(f"{settings['log_dir']}/{dataset_name}/{shot}shot")
    mask = StochasticPixelMask(settings['mask_probability'])
    reconstruction,consistency = MSEReconLoss(),ConsistencyLoss()
    try:
        for step in range(start,settings['steps']):
            pipeline.train()
            sx,sy,qx,qy = (x.to(device) for x in sampler.sample(step))
            # Mask images BEFORE the frozen encoder, as in KT Eq.10.
            sm1,sh1 = pipeline.encode(mask(sx))
            sm2,_ = pipeline.encode(mask(sx))
            qm,qh = pipeline.encode(qx)
            # A random query subset bounds graph memory while retaining all classes over training.
            query_ids = torch.randperm(len(qm),device=device)[:settings['query_batch_size']]
            qm,qh,qy = qm[query_ids],qh[query_ids],qy[query_ids]
            out = head(sm1,qm,sh1,qh,sy,n_way)
            mse = reconstruction(qm,out['reconstructed'])
            cs = consistency(head.projector(out['adapted']),head.adaptation(sm2))
            # Query labels are from the TRAIN split only. CE explicitly trains the voting head.
            classification = F.nll_loss(out['scores'],qy)+F.cross_entropy(out['gnn'],qy)
            entropy = -(out['tau']*out['tau'].clamp_min(1e-12).log()).sum(-1).mean()
            loss = settings['mse_weight']*mse+settings['consistency_weight']*cs+settings['ce_weight']*classification-settings['entropy_weight']*entropy
            optimize(loss,optimizer,parameters)
            scheduler.step()
            if (step+1)%settings['log_every']==0:
                logger.log(step+1,loss=loss.item(),mse=mse.item(),consistency=cs.item(),ce=classification.item(),
                           attention_entropy=entropy.item(),accuracy=(out['scores'].argmax(-1)==qy).float().mean().item())
            improved = False
            if validation and (step+1)%settings.get('validation_every',500)==0:
                score = validate_pipeline(pipeline,validation,device,settings.get('validation_episodes',50))
                logger.log(step+1,validation_accuracy=score)
                improved = score>best
                best = max(best,score)
            if improved or (step+1)%settings['save_every']==0 or step+1==settings['steps']:
                state = snapshot('stage2',step+1,modules,optimizer,scheduler,cfg,dataset=dataset_name,shot=shot,
                                 training_domain=training_name,train_n_way=n_way,
                                 warmup_targets=initial.get('warmup_targets',[]),
                                 initialization_stage=initial['stage'],best_validation_accuracy=best)
                save_checkpoint(state,f'{directory}/last.pt')
                if improved:
                    save_checkpoint(state,f'{directory}/best.pt')
            if stop_after is not None and step+1>=stop_after:
                break
    finally:
        logger.close()
    return f'{directory}/last.pt'

if __name__=='__main__':
    args,cfg = arguments('stage2')
    train(cfg,args.init,args.dataset,args.shot,args.resume)
