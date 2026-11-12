# NoahRGB 11/2026
#
# implements a truncated normal distribution which is used as the output
# distribution of the actor in DreamerV2
#
# adapted from many sources
# https://github.com/pytorch/pytorch/blob/a40812de534b42fcf0eb57a5cecbfdc7a70100cf/torch/nn/init.py#L22
# https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/dists.py#L72
# https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

import torch
import math

SQRT2 = math.sqrt(2.0)

class TruncatedNormalDistribution:

    def __init__(self, mean, std, low=-1.0, high=1.0, eps=1e-6):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
        self.eps = eps

        self.alpha = (low - mean) / std
        self.beta = (high - mean) / std
        self.l = 0.5 * (1 + torch.erf(self.alpha / SQRT2))
        self.u = 0.5 * (1 + torch.erf(self.beta / SQRT2))
        self.Z = (self.u - self.l).clamp_min(1e-8)
        
    def rsample(self, sample_shape=torch.Size()):
        p = torch.rand(torch.Size(sample_shape) + self.mean.shape).to(self.mean.device) 
        y = (2 * self.l - 1) + p * (2 * (self.u - self.l))
        x = torch.erfinv(y.clamp(-1 + self.eps, 1 - self.eps)) * self.std * SQRT2 + self.mean
        clipped = x.clamp(self.low + self.eps, self.high - self.eps)
        return x + (clipped - x).detach() # straight through grad

    def log_prob(self, x):
        prob = torch.distributions.Normal(self.mean, self.std).log_prob(x) - torch.log(self.Z)
        return prob.sum(-1)

    def entropy(self):
        phi = lambda t: torch.exp(-0.5 * t**2) / math.sqrt(2 * math.pi)
        h = (torch.log(math.sqrt(2 * math.pi * math.e) * self.std * self.Z)
                + (self.alpha * phi(self.alpha) - self.beta * phi(self.beta)) / (2 * self.Z))
        return h.sum(-1)