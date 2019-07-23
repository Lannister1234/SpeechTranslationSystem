import numpy as np
import torch
import torch.nn as nn

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, training=False, rate=0.5):
        if rate == 0 or training is False:
            return x
        mask = x.data.new(1, x.size(1), x.size(2))
        mask = mask.bernoulli_(1 - rate)
        mask.requires_grad = False
        mask = mask / (1 - rate)
        mask = mask.expand_as(x)
        x = mask * x
        return x