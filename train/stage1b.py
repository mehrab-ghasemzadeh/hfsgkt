import torch
from torch import nn
from models.full_pipeline import Encoder
from models.cosine_classifier import CosineClassifier
from models.domain_classifier import DomainClassifier
from models.lagrange import LagrangeWeighting
from losses.ce_batch_spectral import BatchSpectralCE
from losses.kl_domain import KLDomainLoss
from losses.ce_domain import CEDomainLoss
from utils.checkpoint import load_checkpoint,save_checkpoint
from utils.logger import TBLogger
from .common import arguments,setup,datasets,sample_batch,optimize,snapshot,restore,save_run_config,verify_metadata


def train(cfg,init,resume=None):
    device = setup(cfg)
    source = datasets(cfg,cfg['source'])['train']
    target_names = [name for name in cfg['stage1b'].get('warmup_targets',cfg['targets']) if cfg['datasets'][name].get('warmup',True)]
    if not target_names:
        raise ValueError('Stage1b needs at least one target warmup training split')
    targets = {name:datasets(cfg,name)['train'] for name in target_names}
    d = cfg['model']['d']
    encoder = Encoder(d).to(device)
    classifier = CosineClassifier(len(source.classes),d).to(device)
    target_heads = nn.ModuleDict({name:CosineClassifier(len(ds.classes),d) for name,ds in targets.items()}).to(device)
    dc_da,dc_ds = DomainClassifier(len(source.classes)).to(device),DomainClassifier(len(source.classes)).to(device)
    weighting = LagrangeWeighting().to(device)
    modules = dict(encoder=encoder,classifier=classifier,target_heads=target_heads,dc_da=dc_da,dc_ds=dc_ds,weighting=weighting)
    settings = cfg['stage1b']
    if not resume:
        initial = load_checkpoint(init,'stage1a')
        verify_metadata(initial,cfg)
        if initial['source_classes']!=source.classes:
            raise ValueError('Source labels differ from stage1a')
        encoder.load_state_dict(initial['models']['encoder'])
        classifier.load_state_dict(initial['models']['classifier'])
    encoder.backbone.requires_grad_(False).eval()
    # Freeze backbone including BN statistics; train hybrid + heads.
    parameters = [p for name,module in modules.items() if name!='dc_da' for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters,lr=settings['lr'],momentum=.9,weight_decay=settings['weight_decay'])
    adversary_optimizer = torch.optim.Adam(dc_da.parameters(),lr=settings['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,settings['step_size'],settings['gamma'])
    spectral = BatchSpectralCE(cfg['loss']['spectral_weight'],cfg['loss']['spectral_k'])
    domain_ce,domain_kl = CEDomainLoss(),KLDomainLoss()
    start = 0
    if resume:
        state = restore(resume,'stage1b',modules,optimizer,scheduler,cfg)
        adversary_optimizer.load_state_dict(state['adversary_optimizer'])
        start = state['step']
        if state['config']['stage1b']!=settings:
            raise ValueError('Resume requires unchanged stage1b settings')
    directory = settings['output']
    save_run_config(cfg,directory)
    logger = TBLogger(settings['log_dir'])
    try:
        for step in range(start,settings['steps']):
            name = target_names[step%len(target_names)]
            xs,ys = sample_batch(source,settings['source_batch_size'],cfg['seed']+2*step,device)
            xt,yt = sample_batch(targets[name],settings['target_batch_size'],cfg['seed']+2*step+1,device)
            with torch.no_grad():
                maps_s,maps_t = encoder.backbone(xs),encoder.backbone(xt)
            ms,hs,_ = encoder.hybrid(*maps_s)
            mt,ht,_ = encoder.hybrid(*maps_t)
            logits_ms,logits_mt = classifier(ms),classifier(mt)
            # Train DA discriminator on true domain labels before confusing features.
            dc_da.requires_grad_(True)
            discriminator_loss = domain_ce(dc_da(logits_ms.detach()),dc_da(logits_mt.detach()))
            optimize(discriminator_loss,adversary_optimizer,list(dc_da.parameters()))
            dc_da.requires_grad_(False)
            fsl_s = .5*(spectral(classifier(hs),hs,ys)+spectral(logits_ms,ms,ys))
            fsl_t = .5*(spectral(target_heads[name](ht),ht,yt)+spectral(target_heads[name](mt),mt,yt))
            alignment = domain_kl(dc_da(logits_ms),dc_da(logits_mt))
            discrimination = domain_ce(dc_ds(classifier(hs)),dc_ds(classifier(ht)))
            total,phi = weighting([fsl_s,fsl_t,alignment,discrimination])
            optimize(total,optimizer,parameters)
            scheduler.step()
            if (step+1)%settings['log_every']==0:
                logger.log(step+1,loss=total.item(),fsl_source=fsl_s.item(),fsl_target=fsl_t.item(),
                           da=alignment.item(),ds=discrimination.item(),adversary=discriminator_loss.item(),
                           **{f'phi_{i}':p.item() for i,p in enumerate(phi)})
            if (step+1)%settings['save_every']==0 or step+1==settings['steps']:
                state = snapshot('stage1b',step+1,modules,optimizer,scheduler,cfg,
                                 adversary_optimizer=adversary_optimizer.state_dict(),source_classes=source.classes,warmup_targets=target_names)
                save_checkpoint(state,f'{directory}/last.pt')
    finally:
        logger.close()
    return f'{directory}/last.pt'

if __name__=='__main__':
    args,cfg = arguments('stage1b')
    train(cfg,args.init,args.resume)
