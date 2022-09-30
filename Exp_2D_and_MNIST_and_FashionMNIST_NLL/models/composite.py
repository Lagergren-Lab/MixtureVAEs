import numpy as np
import torch
from torch.distributions import Normal
from models.PriorNet import VampNet
import torch.nn as nn
from models.VAEIAF import EnsembleIAFs
from models.encoders import EnsembleGatedConv2dEncodersUpper, EnsembleGatedConv2dEncodersLower
from models.decoders import PixelCNNDecoder


class Composite(nn.Module):
    def __init__(self, S=2, L=1, K=500, device='cuda:0', seed=0, x_dims=784,
                 z_dims=40, beta=1., lr=1e-3, T=2):
        super(Composite, self).__init__()
        self.S = S
        self.device = device
        self.seed = seed
        torch.manual_seed(self.seed)
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.obj_f = 'miselbo_beta'
        self.model_name = f"Composite_a_{beta}_seed_{seed}_S_{S}"
        self.K = K
        self.L = L
        self.beta = beta

        self.encoder_2 = EnsembleGatedConv2dEncodersUpper(n_dims=x_dims, latent_dims=z_dims, h=294, S=S,
                                                          device=self.device).to(self.device)
        self.encoder_1 = EnsembleGatedConv2dEncodersLower(n_dims=x_dims, latent_dims=z_dims, h=294,
                                                          S=S, device=self.device).to(self.device)
        self.flow = EnsembleIAFs(device, dim=z_dims, nh=320, T=T, S=S, seed=seed)

        self.vamp_net = VampNet(encoder=self.encoder_2, device=device,
                                n_dims=self.x_dims, K=self.K, convs=True).to(self.device)
        self.phi = list(self.encoder_2.parameters()) + \
                   list(self.encoder_1.parameters()) + \
                   list(self.vamp_net.pseudo_generator.parameters())  + \
                   list(self.flow.parameters())

        self.decoder = PixelCNNDecoder(x_dims, z_dims, h_dim_1=300, h_dim_2=300, device=device).to(self.device)
        self.theta = self.decoder.parameters()
        self.optim = torch.optim.Adam(params=self.phi + list(self.theta), lr=lr, weight_decay=0)

    def sample(self, mu, std, L):
        bs = mu.size(0)
        latent_dims = mu.size(-1)
        expanded_shape = (L, bs, self.S, latent_dims)
        eps = torch.randn(expanded_shape).to(self.device)
        mu = mu.unsqueeze(0).expand(expanded_shape)
        std = std.unsqueeze(0).expand(expanded_shape)
        sample = mu + (eps * std)
        return sample.float()

    def forward(self, x, L=0):
        if L == 0:
            L = self.L
        mu2, std2 = self.encoder_2(x)
        z2 = self.sample(mu2, std2, L)
        mu1, std1 = self.encoder_1(x, z2)
        # only sample z1 wrt s, i.e. z1_s \sim N( |mu1_s(z2_s), std1_s(z2_s)). Diagonal gets mu1_s(z_s) for all s
        idx = torch.arange(mu1.size(1))
        z1 = self.sample(mu1[:, idx, idx, :], std1[:, idx, idx, :], L)
        zT = torch.zeros((L, z1.size(1), self.S, self.S, self.z_dims), device=self.device)
        log_Q = torch.zeros((z1.size(0), z1.size(1), self.S, self.S)).to(self.device)
        for s in range(self.S):
            zT[:, :, s, :, :], log_Q[:, :, s, :] = self.flow(z1[..., s, :])
        recon, p_mu, p_std = self.decoder(x, zT[:, :, idx, idx, :], z2)
        return z1, mu1, std1, z2, mu2, std2, recon, p_mu, p_std, zT, log_Q

    def get_log_Q2(self, z2, mu2, std2):
        # L, bs, S, J, where J=S. log_Q[..., s, :] indexes all S phi_j
        log_Q = torch.zeros((z2.size(0), z2.size(1), self.S, self.S)).to(self.device)

        # mu has dims 1, bs, S, z_dims
        Q_mixt = Normal(mu2.unsqueeze(0), std2.unsqueeze(0))  # N(|mu2_s(x))

        log_pz = torch.zeros((z2.size(0), z2.size(1), self.S)).to(self.device)
        for s in range(self.S):
            # get z from component s and expand to fit Q_mixt dimensions
            z2_s = z2[..., s, :].view((z2.size(0), z2.size(1), 1, z2.size(-1)))
            log_Q[..., s, :] = Q_mixt.log_prob(z2_s).sum(dim=-1)  # log N(z_s|mu2_j, std2_j) for all j
            log_pz[..., s] = self.compute_prior(z2_s)
        return log_Q, log_pz

    def get_log_Q(self, z1, mu1, std1, z2, mu2, std2, log_det):
        logQ2, log_pz2 = self.get_log_Q2(z2, mu2, std2)
        for s in range(self.S):
            mu1_s = mu1[:, :, s, ...]  # mu_j(z_s) for all j
            std1_s = std1[:, :, s, ...]
            Q_mixt = Normal(mu1_s, std1_s)  # N( | mu_j(z2_s), std_j(z2_s))
            z1_s = z1[..., s, :].view((z1.size(0), z1.size(1), 1, z1.size(-1)))
            # compute likelihood of z_s according to the variational ensemble
            logQ2[..., s, :] += Q_mixt.log_prob(z1_s).sum(dim=-1) + log_det[..., s, :]
            # above: log N(z_s|mu2_j, std2_j) + log N(z1_s| mu_j(z2_s), std_j(z2_s)) for all j
        log_Q = torch.logsumexp(logQ2 - np.log(self.S), dim=-1)  # sum the mixture over j wrt z_s
        return log_Q, log_pz2

    def get_log_w(self, x, z1, mu1, std1, z2, mu2, std2, recon, p_mu, p_std, zT, log_det):
        # z has dims L, bs, S, z_dims
        L, bs = z2.size(0), x.size(0)
        x = x.view((1, bs, 1, 784))
        idx = torch.arange(mu1.size(1))

        log_px_z = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)
        log_pz1 = torch.sum(Normal(p_mu, p_std).log_prob(zT[:, :, idx, idx, :]), dim=-1)
        log_Q, log_pz2 = self.get_log_Q(z1, mu1, std1, z2, mu2, std2, log_det)

        log_p = log_px_z + log_pz1 + log_pz2
        log_w = log_px_z + self.beta * (log_pz1 + log_pz2 - log_Q)
        return log_w, log_p, log_Q

    def compute_prior(self, z, fast=True):
        p_mu, p_std = self.vamp_net()
        if fast:
            shape_prior_params = (1, 1, self.K, self.S, self.z_dims)
            shape_prior_z_s = (z.size(0), z.size(1), 1, 1, self.z_dims)
            Pz = Normal(p_mu.view(shape_prior_params), p_std.view(shape_prior_params))
            log_pz = torch.logsumexp(Pz.log_prob(z.view(shape_prior_z_s)).sum(dim=-1) - np.log(self.K), dim=-2)
        else:
            log_pz = torch.logsumexp(torch.stack([
                Normal(m, s).log_prob(z).sum(dim=-1) - np.log(self.K)
                for m, s in zip(p_mu, p_std)
            ]), dim=0)
        log_pz = torch.logsumexp(log_pz - np.log(self.S), dim=-1)
        return log_pz

    def loss(self, log_w, log_p=None, log_q=None, L=0, obj_f='elbo'):
        if obj_f == 'miselbo_beta':
            beta_obj = log_w.mean(dim=-1).sum()
            return - beta_obj
        elif obj_f == "miselbo":
            return - torch.sum(torch.mean(torch.logsumexp(log_p - log_q - np.log(L), dim=0), dim=-1))

    def backpropagate(self, x):
        outputs = self.forward(x)
        log_w, log_p, log_q = self.get_log_w(x, *outputs)

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= x.size(0)  # divide by batch size
        loss.backward()

        # take step
        self.optim.step()

        # reset gradients
        self.optim.zero_grad()
        return loss
