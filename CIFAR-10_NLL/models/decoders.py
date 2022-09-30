import torch
import torch.nn as nn
from models.fc_nets import GatedDense
import pdb
from models.conv_nets import MaskedConv2d, Conv2d


class CNNDecoder(nn.Module):
    def __init__(self, n_dims, no_channels, latent_dims, device="cuda"):

        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.no_channels = no_channels
        self.latent_dims = latent_dims


        self.latent_intermediate = latent_dims * ( (n_dims // 8) ** 2)

        self.decode = nn.Sequential(
            nn.Linear(latent_dims, self.latent_intermediate),
            nn.ReLU(),
        )


        self.decoder_x = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dims, latent_dims // 2,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(latent_dims // 2),
            nn.ReLU(),

            nn.ConvTranspose2d(
                latent_dims//2, latent_dims // 4,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(latent_dims // 4),
            nn.ReLU(),

            nn.ConvTranspose2d(
                latent_dims//4, no_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(no_channels),
            nn.ReLU(),
            nn.Sigmoid()
        )



    def forward(self, z):

        z = self.decode(z.squeeze()).view(-1, self.latent_dims, self.n_dims//8, self.n_dims//8,)

        recon = self.decoder_x(z)

        return recon


class CNNDecoderLogMix(nn.Module):
    def __init__(self, n_dims, no_channels, latent_dims, n_log_mixtures=3, device="cuda"):

        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.no_channels = no_channels
        self.latent_dims = latent_dims
        self.n_log_mixtures = n_log_mixtures

        self.latent_intermediate = latent_dims * ( (n_dims // 8) ** 2)

        self.decode = nn.Sequential(
            nn.Linear(latent_dims, self.latent_intermediate),
            nn.ReLU(),
        )


        self.decoder_x = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dims, latent_dims // 2,
                kernel_size=4, stride=2, padding=1,
                             ),
            nn.BatchNorm2d(latent_dims // 2),
            nn.ReLU(),

            nn.ConvTranspose2d(
                latent_dims//2, latent_dims // 4,
                kernel_size=4, stride=2, padding=1,
                ),
            nn.BatchNorm2d(latent_dims // 4),
            nn.ReLU(),

            nn.ConvTranspose2d(
                latent_dims//4, no_channels,
                kernel_size=4, stride=2, padding=1,
                ),
            nn.BatchNorm2d(no_channels),
            nn.ELU(),
        )

        self.out_features = n_log_mixtures * 10
        # self.log_mixture_layer = nn.Sequential(
        #     nn.Linear(in_features= no_channels, out_features=self.out_features)
        # )

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

        self.log_mixture_layer = nn.Sequential(
            nn.Conv2d(64, self.out_features, kernel_size=3, padding=1, bias=True)
        )



    def forward(self, z):


        z = self.decode(z.squeeze()).view(-1, self.latent_dims, self.n_dims//8, self.n_dims//8,)


        z = self.decoder_x(z)

        z = self.pixelcnn(z)

        log_mix_params = self.log_mixture_layer(z)

        return log_mix_params

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


class TwoLayerGaussDecoder(nn.Module):
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

        self.mu_enc = nn.Linear(in_features=h_dim_1, out_features=n_dims)
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=h_dim_1, out_features=n_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, z):
        z = self.fc_layers(z)
        mu = self.mu_enc(z)
        std = torch.exp(0.5 * self.log_var_enc(z))  # 0.01 + func.softplus(self.std_enc(x))
        return mu, std





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
