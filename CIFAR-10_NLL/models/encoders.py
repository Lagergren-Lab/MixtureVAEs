import torch.nn as nn
import torch.nn.functional as func
import torch
from models.fc_nets import GatedDense
from torch.autograd import Variable


class CNNEncoder(nn.Module):
    def __init__(self, n_dims, no_channels, latent_dims, device='cuda:0'):

        super().__init__()

        self.n_dims = n_dims
        self.no_channels = no_channels
        self.latent_dims = latent_dims
        self.device = device


        self.encoder = nn.Sequential(
            nn.Conv2d(
                no_channels, latent_dims // 4,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(latent_dims // 4),
            nn.ReLU(),
            nn.Conv2d(
                latent_dims // 4, latent_dims // 2,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(latent_dims // 2),
            nn.ReLU(),

            nn.Conv2d(
                latent_dims // 2, latent_dims,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(latent_dims),
            nn.ReLU(),
        )

        self.latent_intermediate = latent_dims * ( (n_dims // 8) ** 2)

        self.mu_encoder = nn.Linear(self.latent_intermediate, latent_dims)
        self.logvar_encoder = nn.Sequential(
            nn.Linear(self.latent_intermediate, latent_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, x):

        x = self.encoder(x).view(-1, self.latent_intermediate)

        mu, logvar = self.mu_encoder(x), self.logvar_encoder(x)

        return mu, torch.exp(0.5 * logvar)


class TwoLayerEncoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0', gated=True):
        super().__init__()
        self.device = device
        self.activation = activation
        if not gated:
            self.fc_layers = nn.Sequential(
                nn.Linear(in_features=n_dims, out_features=h_dim_1),
                self.activation(),
                nn.Linear(in_features=h_dim_1, out_features=h_dim_2),
                self.activation(),

            )
        else:
            self.fc_layers = nn.Sequential(
            GatedDense(n_dims, h_dim_1),
            GatedDense(h_dim_1, h_dim_2)
        )
        self.mu_enc = nn.Linear(in_features=h_dim_2, out_features=latent_dims)
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=h_dim_2, out_features=latent_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        mu = self.mu_enc(x)
        std = torch.exp(0.5 * self.log_var_enc(x))  # 0.01 + func.softplus(self.std_enc(x))
        return mu, std


class EnsembleEncoders(nn.Module):
    def __init__(self, n_dims, latent_dims, S=2,no_channels=3, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0',
                 gated=True):
        super().__init__()
        self.device = device
        self.S = S
        self.latent_dims = latent_dims
        self.encoders = nn.ModuleList([
            CNNEncoder(n_dims = n_dims, latent_dims = latent_dims, no_channels=no_channels,
                            device=device) for _ in range(self.S)]
        )

    def forward(self, x):
        mu = torch.zeros((x.size(0), self.S, self.latent_dims), device=self.device)
        std = torch.zeros_like(mu)

        for s, encoder in enumerate(self.encoders):
            mu[:, s, :], std[:, s, :] = encoder(x)
        return mu, std


class TwoLayerwDropoutEncoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0', gated=True):
        super().__init__()
        self.device = device
        self.activation = activation
        if not gated:
            self.fc_layers = nn.Sequential(
                nn.Linear(in_features=n_dims, out_features=h_dim_1),
                self.activation(),
                nn.Linear(in_features=h_dim_1, out_features=h_dim_2),
                self.activation(),

            )
        else:
            self.fc_layers = nn.Sequential(
                            GatedDense(n_dims, h_dim_1),
                            nn.Dropout(0.2),
                            GatedDense(h_dim_1, h_dim_2),
                            nn.Dropout(0.2),
                                            )
        self.mu_enc = nn.Linear(in_features=h_dim_2, out_features=latent_dims)
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=h_dim_2, out_features=latent_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        mu = self.mu_enc(x)
        std = torch.exp(0.5 * self.log_var_enc(x))  # 0.01 + func.softplus(self.std_enc(x))
        return mu, std


class DropoutEnsembleEncoders(nn.Module):
    def __init__(self, n_dims, latent_dims, S=2, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0',
                 gated=True):
        super().__init__()
        self.device = device
        self.S = S
        self.latent_dims = latent_dims
        self.encoder = TwoLayerwDropoutEncoder(n_dims, latent_dims, h_dim_1=h_dim_1, h_dim_2=h_dim_2,
                            activation=activation, device='cuda:0', gated=gated)

    def forward(self, x):
        self.train()  # make sure to always use dropout actively
        mu = torch.zeros((x.size(0), self.S, self.latent_dims), device=self.device)
        std = torch.zeros_like(mu)
        for s in range(self.S):
            mu[:, s, :], std[:, s, :] = self.encoder(x)
        return mu, std


if __name__ == '__main__':
    enc = DropoutEnsembleEncoders(5, 2, S=2)
    x = torch.ones((1, 5))
    mu, std = enc(x)
    print(mu)
