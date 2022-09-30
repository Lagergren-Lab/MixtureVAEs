import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from models.encoders import EnsembleEncoders, DropoutEnsembleEncoders, EnsembleGatedConv2dEncodersUpper
from models.decoders import SingleLayerPixelCNNDecoder
from models.PriorNet import VampNet, GMMNet
from models.vaes import VAE


class MISVAE(VAE):
    def __init__(self, S=2, decoder=None, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3, ais=False):
        super().__init__(decoder=decoder, L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device)
        self.S = S
        self.device = device
        if ais:
            self.encoder = None
        else:
            self.encoder = EnsembleEncoders(n_dims=x_dims, latent_dims=z_dims, h_dim_1=300, h_dim_2=300,
                                            S=S, device=self.device, gated=True).to(self.device)
            self.phi = self.encoder.parameters()
            self.theta = self.decoder.parameters()
            self.optim = torch.optim.Adam(params=list(self.phi) + list(self.theta), lr=lr, weight_decay=0)

        self.model_name = f"MISVAE_a_{beta}_seed_{seed}_S_{S}"
        self.obj_f = 'miselbo_beta'

        # number of importance samples
        self.L = L
        self.beta = beta

    def sample(self, mu, std, L):
        bs = mu.size(0)
        latent_dims = mu.size(-1)
        expanded_shape = (L, bs, self.S, latent_dims)
        eps = torch.randn(expanded_shape).to(self.device)
        mu = mu.unsqueeze(0).expand(expanded_shape)
        std = std.unsqueeze(0).expand(expanded_shape)
        sample = mu + (eps * std)
        return sample.float()

    def loss(self, log_w, log_p=None, log_q=None, L=0, obj_f='elbo'):
        if L == 0:
            L = self.L
        if obj_f == 'elbo':
            elbo = log_w.sum()
            return - elbo
        elif obj_f == 'iwelbo':
            return - torch.sum(torch.logsumexp(log_p - log_q - np.log(L), dim=0))
        elif obj_f == 'miselbo_beta':
            beta_obj = log_w.mean(dim=-1).sum()
            return - beta_obj
        elif obj_f == "miselbo":
            return - torch.sum(torch.mean(torch.logsumexp(log_p - log_q - np.log(L), dim=0), dim=-1))

    def backpropagate(self, x):
        z, mu, std, recon = self.forward(x)
        log_w, log_p, log_q = self.get_log_w(x, z, mu, std, recon)

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= mu.size(0)  # divide by batch size
        loss.backward()

        # take step
        self.optim.step()

        # reset gradients
        self.optim.zero_grad()
        return loss

    def get_log_w(self, x, z, mu, std, recon, return_jsd=False):
        # z has dims L, bs, S, z_dims
        L, bs = z.size(0), x.size(0)
        x = x.view((1, bs, 1, 784))

        log_px_z = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)

        # z has dims L, bs, S, z_dims
        log_Q = torch.zeros((z.size(0), z.size(1), self.S)).to(self.device)
        if return_jsd:
            log_Q_s = torch.zeros_like(log_Q)
        # mu has dims 1, bs, S, z_dims
        Q_mixt = Normal(mu, std)

        log_pz = torch.zeros_like(log_Q)
        for s in range(self.S):
            # get z from component s and expand to fit Q_mixt dimensions
            z_s = z[..., s, :].view((z.size(0), z.size(1), 1, z.size(-1)))
            # compute likelihood of z_s according to the variational ensemble
            if return_jsd:
                log_Q_mixt_non_summed = Q_mixt.log_prob(z_s).sum(dim=-1)
                log_Q_s[..., s] = log_Q_mixt_non_summed[..., s]
                log_Q_mixture_wrt_z_s = torch.logsumexp(log_Q_mixt_non_summed - np.log(self.S), dim=-1)
            else:
                log_Q_mixture_wrt_z_s = torch.logsumexp(Q_mixt.log_prob(z_s).sum(dim=-1) - np.log(self.S), dim=-1)
            log_Q[..., s] = log_Q_mixture_wrt_z_s
            log_pz[..., s] = self.compute_prior(z_s)

        log_p = log_px_z + log_pz
        log_w = log_px_z + self.beta * (log_pz - log_Q)
        if return_jsd:
            return log_w, log_p, log_Q, log_Q_s
        else:
            return log_w, log_p, log_Q

    def compute_prior(self, z):
        z = z.squeeze(-2)
        return Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)


class MISVAEwVamp(MISVAE):
    def __init__(self, S=2, K=500, vamp_net=None, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3):
        super().__init__(S=S, L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device)

        self.model_name = f"MISVAEwVamp_a_{beta}_seed_{seed}_S_{S}"
        self.obj_f = 'miselbo_beta'

        # number of importance samples
        self.L = L
        self.beta = beta
        self.K = K

        if vamp_net is not None:
            # parameters in vamp net are not trainable
            self.vamp_net = vamp_net
            self.K = self.vamp_net.K
            self.phi = list(self.encoder.parameters())
        else:
            # both vamp net and encoder net params are
            self.vamp_net = VampNet(encoder=self.encoder, device=device, n_dims=self.x_dims, K=self.K).to(self.device)
            self.phi = list(self.encoder.parameters()) + list(self.vamp_net.pseudo_generator.parameters())

        self.theta = self.decoder.parameters()
        self.optim = torch.optim.Adam(params=self.phi + list(self.theta), lr=lr, weight_decay=0)

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


class MISVAECNN(MISVAE):
    def __init__(self, S=2, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3):
        super().__init__(S=S, L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device)
        self.model_name = f"MISVAEwCNN_a_{beta}_seed_{seed}_S_{S}"
        self.obj_f = 'miselbo_beta'

        # number of importance samples
        self.L = L
        self.beta = beta

        self.encoder = EnsembleGatedConv2dEncodersUpper(n_dims=x_dims, latent_dims=z_dims, h=294, S=S,
                                                        device=self.device).to(self.device)
        self.decoder = SingleLayerPixelCNNDecoder(x_dims, z_dims, h_dim_1=300, h_dim_2=300, device=device
                                                  ).to(self.device)

        self.phi = list(self.encoder.parameters())
        self.theta = self.decoder.parameters()
        self.optim = torch.optim.Adam(params=list(self.phi) + list(self.theta), lr=lr, weight_decay=0)

    def forward(self, x, L=0):
        if L == 0:
            L = self.L
        mu, std = self.encoder(x)
        z = self.sample(mu, std, L)
        reconstruction = self.decoder(x, z)
        return z, mu, std, reconstruction


class MISVAECNNwVamp(MISVAECNN):
    def __init__(self, S=2, K=500, vamp_net=None, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3):
        super().__init__(S=S, L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device, beta=beta)

        self.K = K
        if vamp_net is not None:
            # parameters in vamp net are not trainable
            self.vamp_net = vamp_net
            self.K = self.vamp_net.K
            self.phi = list(self.encoder.parameters())
        else:
            # both vamp net and encoder net params are
            self.vamp_net = VampNet(encoder=self.encoder, device=device, n_dims=self.x_dims,
                                    K=self.K, convs=True).to(self.device)
            self.phi = list(self.encoder.parameters()) + list(self.vamp_net.pseudo_generator.parameters())

        self.theta = self.decoder.parameters()
        self.optim = torch.optim.Adam(params=self.phi + list(self.theta), lr=lr, weight_decay=0)

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


class MISVAEwGMM(MISVAE):
    def __init__(self, S=2, K=500, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3):
        super().__init__(S=S, L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device)

        self.model_name = f"MISVAEwGMM_a_{beta}_seed_{seed}_S_{S}"

        # number of importance samples
        self.L = L
        self.beta = beta
        self.K = K

        # both vamp net and encoder net params are
        self.vamp_net = GMMNet(n_dims=self.x_dims, latent_dims=z_dims, K=self.K)
        self.phi = list(self.encoder.parameters()) + list(self.vamp_net.parameters())

        self.theta = self.decoder.parameters()
        self.optim = torch.optim.Adam(params=self.phi + list(self.theta), lr=lr, weight_decay=0)


class DropMISVAE(MISVAE):
    def __init__(self, S=2, train_pi=False, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3):
        super().__init__(S=S, train_pi=train_pi, L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device)

        self.encoder = DropoutEnsembleEncoders(n_dims=x_dims, latent_dims=z_dims, S=S, gated=True).to(self.device)

        self.model_name = f"DropMISVAE_a_{beta}_seed_{seed}_S_{S}"
        self.phi = self.encoder.parameters()
        self.theta = self.decoder.parameters()

        self.optim = torch.optim.Adam(params=list(self.phi) + list(self.theta), lr=lr, weight_decay=0)

        self.obj_f = 'miselbo_beta'


"""
    def sample(self, mu, std, L):
        bs = mu.size(0)
        latent_dims = mu.size(-1)
        expanded_shape = (L, bs, self.S, latent_dims)
        eps = torch.randn(expanded_shape).to(self.device)
        mu = mu.unsqueeze(0).expand(expanded_shape)
        std = std.unsqueeze(0).expand(expanded_shape)
        sample = mu + (eps * std)
        return sample.float()

    def get_log_w(self, x, z, mu, std, recon):
        # z has dims L, bs, S, z_dims
        L, bs = z.size(0), z.size(1)
        x = x.view((1, bs, 1, x.size(-1)))

        log_px_z = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)

        # z has dims L, bs, S, z_dims
        log_Q = torch.zeros((L, bs, self.S)).to(self.device)
        # mu has dims 1, bs, S, z_dims
        Q_mixt = Normal(mu, std)
        log_pz = torch.zeros_like(log_Q)
        for s in range(self.S):
            # get z from component s and expand to fit Q_mixt dimensions
            z_s = z[..., s, :].view((z.size(0), z.size(1), 1, z.size(-1)))
            # compute likelihood of z_s according to the variational ensemble
            log_Q_mixture_wrt_z_s = torch.logsumexp(Q_mixt.log_prob(z_s).sum(dim=-1) - np.log(self.S), dim=-1)
            log_Q[..., s] = log_Q_mixture_wrt_z_s
            log_pz[..., s] = self.compute_prior(z_s)

        log_p = log_px_z + log_pz
        log_w = log_px_z + self.beta * (log_pz - log_Q)
        return log_w, log_p, log_Q

"""



