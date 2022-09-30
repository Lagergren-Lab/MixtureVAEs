import torch.nn as nn


class GatedDense(nn.Module):
    # from https://github.com/jmtomczak/vae_vampprior/blob/master/utils/nn.py
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g















