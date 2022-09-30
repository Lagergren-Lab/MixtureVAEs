"""
source: https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/nets.py
Various helper network modules
"""

from torch import nn
from .MADE import MADE


class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh], nout, num_masks=1, natural_ordering=True)

    def forward(self, x):
        return self.net(x)
