import torch
from torch import nn
from .networks import ARMLP
from models.encoders import TwoLayerEncoder
from models.decoders import TwoLayerDecoder, OneLayerDecoder
from torch.distributions import Normal


class EnsembleIAFs(nn.Module):
    def __init__(self, device, dim, nh, arg_class=ARMLP, T=3, S=1, seed=0):
        super().__init__()
        self.device = device
        self.dim = dim
        self.nh = nh  # n hidden units in made
        self.S = S
        self.flow_ensemble = nn.ModuleList([
            NIAFEncoder(device=device, seed=seed, T=T, nh=nh, arg_class=arg_class, dim=dim).to(self.device)
            for _ in range(S)
        ])

    def forward(self, z_s):
        # input s'th sample and run it thru iaf for all j
        zT_sj = torch.zeros((z_s.shape[0], z_s.shape[1], self.S, z_s.shape[-1]), device=self.device)
        log_detT_sj = torch.zeros((z_s.shape[0], z_s.shape[1], self.S), device=self.device)
        for j, iaf in enumerate(self.flow_ensemble):
            zT_sj[..., j, :], log_detT_sj[..., j] = iaf.encode(z_s)
        return zT_sj, log_detT_sj


class NIAFEncoder(nn.Module):

    def __init__(self, device, dim, nh=24, arg_class=ARMLP, T=3, seed=0):
        super().__init__()
        self.dim = dim
        torch.manual_seed(seed)
        mades = [arg_class(dim, dim*2, nh) for t in range(T)]
        self.flows = nn.ModuleList(mades)
        self.device = device

    def encode(self, z):

        log_det = torch.zeros((z.size(0), z.size(1)), device=self.device)

        for flow in self.flows:
            sm = flow.forward(z)
            s, m = sm.split(self.dim, dim=-1)

            std = torch.sigmoid(s)
            z = std*z + (1-std) * m
            log_det += -torch.sum(torch.log(std), dim=-1)

        return z, log_det


class IAFVAE(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, device, latent_dims, data_dims=28*28, h_dim_1=500, h_dim_2 = 300):
        super().__init__()
        self.flow = NIAFEncoder(device, dim = latent_dims)
        self.decoder = TwoLayerDecoder(n_dims=data_dims, latent_dims=latent_dims)
        self.encoder = TwoLayerEncoder(n_dims=data_dims, latent_dims=latent_dims, h_dim_1=h_dim_1, h_dim_2=h_dim_2)
        self.model_name = ""
        self.device = device


    def reparameterize(self, mu_z, std):

        eps = torch.randn_like(std)

        z = eps * std + mu_z

        return z

    def forward(self, x):
        
        mu, std = self.encoder(x.float())
        z = self.reparameterize(mu, std)

        log_qZ0_x = Normal(mu, std).log_prob(z).sum(dim=-1)

        zs, log_det = self.flow.encode(z)
        zT = zs[-1]

        log_qZT_x = log_qZ0_x + log_det

        log_pZT = Normal(torch.zeros_like(zT), torch.ones_like(zT)).log_prob(zT).sum(dim=-1)

        recon = self.decoder(zT)
        log_px_zT = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)

        ELBO = log_px_zT + log_pZT - log_qZT_x

        return ELBO