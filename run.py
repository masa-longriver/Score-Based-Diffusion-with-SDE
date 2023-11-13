import numpy as np
import torch
import torch.nn.functional as F

def score_fn(x, t, model, sde):
    labels = t * 999
    score = model(x, labels)
    std = sde.marginal_prob(torch.zeros_like(x), t)[1]
    score = -score / std[:, None, None, None]

    return score

def loss_fn(config, x, model, sde):
    t = (
        torch.rand(x.shape[0], device=x.device) 
        * (config['sde']['T'] - config['sde']['eps']) 
        + config['sde']['eps']
    )
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    perturbed_x = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_x, t, model, sde)
    losses = torch.square(score * std[:, None, None, None] + z)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)

    return loss
    
def run(config, epoch, model, sde, dl, optim, ema, state='train'):
    running_loss = 0.

    if state == 'train':
        model.train()
        for x, _ in dl:
            x = x.to(config['device'])
            optim.zero_grad()
            loss = loss_fn(config, x, model, sde)
            loss.backward()
            for g in optim.param_groups:
                g['lr'] = (
                    config['optim']['lr'] 
                    * np.minimum(epoch / config['optim']['warmup'], 1.0)
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['optim']['grad_clip']
            )
            optim.step()
            ema.update()
            running_loss += loss.item() * x.size(0)
    
    elif state == 'eval':
        ema.apply_shadow()
        model.eval()
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(config['device'])
                loss = loss_fn(config, x, model, sde)
                running_loss += loss.item() * x.size(0)
        ema.restore()
    
    return running_loss / len(dl)