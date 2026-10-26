"""Validation uses held-out manifests; never selects weights using the test set."""
import torch

@torch.no_grad()
def validate_pipeline(pipeline,sampler,device,n_episodes):
    was_training = pipeline.training
    pipeline.eval()
    values = []
    try:
        for episode in range(n_episodes):
            sx,sy,qx,qy = (t.to(device) for t in sampler.sample(episode))
            scores = pipeline(sx,qx,sy,sampler.n_way)
            values.append((scores.argmax(-1)==qy).float().mean().item())
    finally:
        pipeline.train(was_training)
    return sum(values)/len(values)

@torch.no_grad()
def validate_encoder(encoder,sampler,device,n_episodes):
    was_training = encoder.training
    encoder.eval()
    values = []
    try:
        for episode in range(n_episodes):
            sx,sy,qx,qy = (t.to(device) for t in sampler.sample(episode))
            _,support,_ = encoder(sx)
            _,queries,_ = encoder(qx)
            prototypes = torch.stack([support[sy==c].mean(0) for c in range(sampler.n_way)])
            prediction = torch.cdist(queries,prototypes).argmin(-1)
            values.append((prediction==qy).float().mean().item())
    finally:
        encoder.train(was_training)
    return sum(values)/len(values)
