import numpy as np
import torch

from run import score_fn

class EulerMaruyamaMethod():
    def __init__(self, config):
        self.config = config

    def step(self, x, t, model, sde):
        dt = -1. / self.config['sde']['timesteps']
        z = torch.randn_like(x)
        score = score_fn(x, t, model, sde)
        drift, diffusion = sde.reverse().sde(x, t, score)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z

        return x, x_mean

def sampling(config, shape, model, sde):
    sampler = EulerMaruyamaMethod(config)
    with torch.no_grad():
        x = sde.prior_sampling(shape).to(config['device'])
        timesteps = torch.linspace(
            config['sde']['T'],
            config['sde']['eps'],
            config['sde']['timesteps'],
            device=config['device']
        )

        for i in range(config['sde']['timesteps']):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            x, x_mean = sampler.step(x, vec_t, model, sde)
    
    return x_mean