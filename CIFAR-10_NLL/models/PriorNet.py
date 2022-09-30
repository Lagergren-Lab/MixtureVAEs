import torch
import torch.nn as nn


class VampNet(nn.Module):
    def __init__(self,
                 encoder,
                 n_dims,
                 K=2,
                 activation=nn.Hardtanh(min_val=0.0, max_val=1.0),
                 device='cuda:0'):
        super().__init__()

        self.n_dims = n_dims
        self.K = K
        self.device = device
        self.activation = activation
        self.encoder = encoder

        self.idle_input = torch.autograd.Variable(torch.eye(self.K, self.K), requires_grad=False).to(self.device)

        self.pseudo_generator = nn.Sequential(
            nn.Linear(in_features=self.K, out_features=self.n_dims), self.activation,
        ).to(self.device)

    def forward(self):
        # get K x n_dims pseudo inputs
        pseudo_inputs = self.pseudo_generator(self.idle_input)
        # prior components' parameters, phi
        vamp_mu, vamp_std = self.encoder(pseudo_inputs)
        return vamp_mu, vamp_std


class GMMNet(nn.Module):
    def __init__(self,
                 n_dims,
                 latent_dims,
                 K=2,
                 activation=nn.ReLU(),
                 device='cuda:0'):
        super().__init__()

        self.n_dims = n_dims
        self.K = K
        self.device = device
        self.activation = activation
        self.latent_dims = latent_dims

        self.idle_input = torch.autograd.Variable(torch.eye(self.n_dims, self.n_dims), requires_grad=False).to(self.device)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=self.n_dims, out_features=self.latent_dims), self.activation,
        ).to(self.device)
        self.mu_encs = nn.ModuleList([
            nn.Linear(in_features=latent_dims, out_features=latent_dims) for _ in range(self.K)
        ])
        self.log_var_encs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=latent_dims, out_features=latent_dims),
                nn.Hardtanh(min_val=-6., max_val=2.)
            ) for _ in range(self.K)
        ])

    def forward(self):
        x = self.fc_layer(self.idle_input)
        mu = torch.zeros((self.K, self.latent_dims), device=self.device)
        std = torch.zeros_like(mu)
        for k in range(self.K):
            mu[k, :], std[k, :] = self.mu_encs[k](x), self.log_var_encs[k](x)
        return mu, std

