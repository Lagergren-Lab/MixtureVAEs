import torch
import torch.nn as nn
from models.fc_nets import GatedDense
from models.conv_nets import MaskedConv2d, Conv2d


class TwoLayerDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0', gated=True):
            super().__init__()
            self.device = device
            self.activation = activation
            if not gated:
                self.fc_layers = nn.Sequential(
                    nn.Linear(in_features=latent_dims, out_features=h_dim_2),
                    self.activation(),
                    nn.Linear(in_features=h_dim_2, out_features=h_dim_1),
                    self.activation(),

                )
            else:
                self.fc_layers = nn.Sequential(
                    GatedDense(latent_dims, h_dim_2),
                    GatedDense(h_dim_2, h_dim_1)
                )
            self.bernoulli_dec = nn.Linear(in_features=h_dim_1, out_features=n_dims)

    def forward(self, z):
        x = self.fc_layers(z)
        return torch.sigmoid(self.bernoulli_dec(x))


class OneLayerDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim=20, activation=nn.ReLU, device='cuda:0'):
            super().__init__()
            self.device = device
            self.activation = activation
            self.fc_layer = nn.Sequential(
                            nn.Linear(in_features=latent_dims, out_features=h_dim),
                            self.activation(),
                        )
            self.bernoulli_dec = nn.Linear(in_features=h_dim, out_features=n_dims)

    def forward(self, z):
        x = self.fc_layer(z)
        return torch.sigmoid(self.bernoulli_dec(x))


class PixelCNNDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0'):
            super().__init__()
            self.device = device
            self.activation = activation

            # p(z1|z2)
            self.fc_layers_lower = nn.Sequential(
                GatedDense(latent_dims, h_dim_2),
                GatedDense(h_dim_2, h_dim_1)
            )

            self.mu_z1 = nn.Linear(in_features=h_dim_1, out_features=latent_dims)
            self.log_var_z1 = nn.Sequential(
                nn.Linear(in_features=h_dim_1, out_features=latent_dims),
                nn.Hardtanh(min_val=-6., max_val=2.)
            )

            self.p_x_layers_z1 = nn.Sequential(
                GatedDense(latent_dims, n_dims)
            )
            self.p_x_layers_z2 = nn.Sequential(
                GatedDense(latent_dims, n_dims)
            )

            # PixelCNN
            act = nn.ReLU(True)
            self.pixelcnn = nn.Sequential(
                MaskedConv2d('A', 1 + 2 * 1, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act
            )

            self.bernoulli_dec = Conv2d(64, 1, 1, 1, 0)

    def forward(self, x_in, z1, z2):
        z2_ = self.fc_layers_lower(z2)
        mu = self.mu_z1(z2_)
        std = torch.exp(0.5 * self.log_var_z1(z2_))

        z2 = self.p_x_layers_z2(z2)
        z1 = self.p_x_layers_z1(z1)

        # L, bs, S, 1, 28, 28
        x_out = torch.zeros((z1.size(0), z1.size(1), z1.size(2), 1, 28, 28), device=self.device)
        z1 = z1.view((z1.size(1), z1.size(2), 1, 28, 28))
        z2 = z2.view((z2.size(1), z2.size(2), 1, 28, 28))

        for s in range(x_out.size(2)):
            x = torch.cat((x_in, z1[:, s], z2[:, s]), dim=-3)
            x = self.pixelcnn(x)
            x_out[:, :, s] = torch.sigmoid(self.bernoulli_dec(x))
        x_out = x_out.view((x_out.size(0), x_out.size(1), x_out.size(2), 784))
        return x_out, mu, std


class SingleLayerPixelCNNDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0'):
            super().__init__()
            self.device = device
            self.activation = activation

            self.p_x_layers = nn.Sequential(
                GatedDense(latent_dims, n_dims)
            )

            # PixelCNN
            act = nn.ReLU(True)
            self.pixelcnn = nn.Sequential(
                MaskedConv2d('A', 1 + 1, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act
            )

            self.bernoulli_dec = Conv2d(64, 1, 1, 1, 0)

    def forward(self, x_in, z):
        z = self.p_x_layers(z)

        # L, bs, S, 1, 28, 28
        x_out = torch.zeros((z.size(0), z.size(1), z.size(2), 1, 28, 28), device=self.device)
        z = z.view((z.size(1), z.size(2), 1, 28, 28))

        for s in range(x_out.size(2)):
            x = torch.cat((x_in, z[:, s]), dim=-3)
            x = self.pixelcnn(x)
            x_out[:, :, s] = torch.sigmoid(self.bernoulli_dec(x))
        x_out = x_out.view((x_out.size(0), x_out.size(1), x_out.size(2), 784))
        return x_out



