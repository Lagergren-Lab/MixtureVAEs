import torch.nn as nn
import torch.nn.functional as func
import torch
from models.fc_nets import GatedDense, VanillaDense
from models.conv_nets import GatedConv2d

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


class NLayerEncoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim=300, No_layers=1, activation=nn.ReLU, device='cuda:0'):
        super().__init__()
        self.device = device
        self.activation = activation

        first_layer = VanillaDense(n_dims, h_dim, activation = activation)
        layers = nn.ModuleList([first_layer] + [VanillaDense(h_dim, h_dim, activation = activation) for layer in range(No_layers)])
        self.fc_layers = nn.Sequential(*layers)

        self.mu_enc = nn.Linear(in_features=h_dim, out_features=latent_dims, bias = False)
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=h_dim, out_features=latent_dims, bias = False),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        mu = self.mu_enc(x)
        std = torch.exp(0.5 * self.log_var_enc(x))  # 0.01 + func.softplus(self.std_enc(x))
        return mu, std


class EnsembleEncoders(nn.Module):
    def __init__(self, n_dims, latent_dims, S=2, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0',
                 gated=True, NLayered = True, No_layers = 1):
        super().__init__()
        self.device = device
        self.S = S
        self.latent_dims = latent_dims

        if NLayered:
            self.encoders = nn.ModuleList([
                NLayerEncoder(n_dims, latent_dims, h_dim=h_dim_1, No_layers = No_layers,
                                activation=activation, device=device) for _ in range(self.S)]
            )
        else:
            self.encoders = nn.ModuleList([
                TwoLayerEncoder(n_dims, latent_dims, h_dim_1=h_dim_1, h_dim_2=h_dim_2,
                                activation=activation, device=device, gated=gated) for _ in range(self.S)]
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


class GatedConv2dEncoderUpper(nn.Module):
    def __init__(self, n_dims, latent_dims, h=294, activation=nn.ReLU, device='cuda:0'):
        super().__init__()
        self.device = device
        self.activation = activation
        self.h = h

        self.conv_layer = nn.Sequential(
            GatedConv2d(1, 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )

        self.mu_enc = nn.Linear(in_features=h, out_features=latent_dims)
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=h, out_features=latent_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view((-1, self.h))
        mu = self.mu_enc(x)
        std = torch.exp(0.5 * self.log_var_enc(x))
        return mu, std


class GatedConv2dEncoderLower(nn.Module):
    def __init__(self, n_dims, latent_dims, h=294, activation=nn.ReLU, device='cuda:0'):
        super().__init__()
        self.device = device
        self.activation = activation
        self.n_dims = n_dims
        self.h = h

        self.conv_layer = nn.Sequential(
            GatedConv2d(1, 32, 3, 1, 1),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )

        self.fc_layer_z2 = nn.Sequential(
            GatedDense(latent_dims, h)
        )

        self.fc_layer_z2_and_x = nn.Sequential(
            GatedDense(2 * h, 300)
        )

        self.mu_enc = nn.Linear(in_features=300, out_features=latent_dims)
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=300, out_features=latent_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, x, z2):
        x = self.conv_layer(x)
        x = x.view((1, -1, 1, self.h))
        x = x.repeat((1, 1, z2.size(-2), 1))
        z = self.fc_layer_z2(z2)
        zx = self.fc_layer_z2_and_x(torch.cat((x, z), dim=-1))
        mu = self.mu_enc(zx)
        std = torch.exp(0.5 * self.log_var_enc(zx))
        return mu, std


class EnsembleGatedConv2dEncodersUpper(nn.Module):
    def __init__(self, n_dims, latent_dims, h=294, S=2, activation=nn.ReLU, device='cuda:0'):
        super().__init__()
        self.device = device
        self.S = S
        self.latent_dims = latent_dims
        self.encoders = nn.ModuleList([
            GatedConv2dEncoderUpper(n_dims, latent_dims, h=h, activation=activation, device=device)
            for _ in range(self.S)]
        )

    def forward(self, x):
        mu = torch.zeros((x.size(0), self.S, self.latent_dims), device=self.device)
        std = torch.zeros_like(mu)
        for s, encoder in enumerate(self.encoders):
            mu[:, s, :], std[:, s, :] = encoder(x)
        return mu, std


class EnsembleGatedConv2dEncodersLower(nn.Module):
    def __init__(self, n_dims, latent_dims, h=294, S=2, activation=nn.ReLU, device='cuda:0'):
        super().__init__()
        self.device = device
        self.S = S
        self.latent_dims = latent_dims
        self.encoders = nn.ModuleList([
            GatedConv2dEncoderLower(n_dims, latent_dims, h=h, activation=activation, device=device)
            for _ in range(self.S)]
        )

    def forward(self, x, z2):
        mu = torch.zeros((x.size(0), self.S, self.S, self.latent_dims), device=self.device)
        std = torch.zeros_like(mu)
        for j, encoder in enumerate(self.encoders):
            mu[:, j, :, :], std[:, j, :, :] = encoder(x, z2)  # gets mu_j(z2_s) for all s
        return mu, std


