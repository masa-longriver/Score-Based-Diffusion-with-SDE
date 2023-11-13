import torch
import numpy as np

class VPSDE():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.beta_0 = self.config['sde']['beta_min']
        self.beta_1 = self.config['sde']['beta_max']
    
    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion
    
    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        return mean, std
    
    def prior_sampling(self, shape):

        return torch.randn(*shape)
    
    def reverse(self):
        sde_fn = self.sde

        class ReverseSDE(self.__class__):
            def sde(self, x, t, score):
                drift, diffusion = sde_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score

                return drift, diffusion
        
        return ReverseSDE(self.config)