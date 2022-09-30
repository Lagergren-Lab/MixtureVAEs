import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from models.encoders import TwoLayerEncoder
from models.decoders import TwoLayerDecoder, OneLayerDecoder
from models.PriorNet import VampNet


# Super class for VAEs. Use the VanillaVAE, AlphaVAE or ChiVAE classes instead
class VAE(nn.Module):
    def __init__(self, decoder=None, L=1, device='cuda:0', share_theta=False, seed=0, x_dims=784, z_dims=40):
        super(VAE, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.device = device
        self.share_theta = share_theta
        self.encoder = TwoLayerEncoder(n_dims=x_dims, latent_dims=z_dims, gated=True).to(self.device)
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = TwoLayerDecoder(n_dims=x_dims, latent_dims=z_dims, gated=True).to(self.device)
        # number of importance samples
        self.L = L
        self.alpha = None
        self.beta = 1.

    def sample(self, mu, std, L):
        bs = mu.size(0)
        latent_dims = mu.size(1)
        eps = torch.randn((L, bs, latent_dims)).to(self.device)
        mu = mu.unsqueeze(0).expand((L, bs, latent_dims))
        std = std.unsqueeze(0).expand((L, bs, latent_dims))
        sample = mu + (eps * std)
        return sample.float()

    def forward(self, x, L=0):
        if L == 0:
            L = self.L
        mu, std = self.encoder(x)
        z = self.sample(mu, std, L)
        reconstruction = self.decoder(z)
        return z, mu, std, reconstruction

    def get_log_w(self, x, z, mu, std, recon):
        log_px_z = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)
        log_pz = Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
        log_q = Normal(mu, std).log_prob(z).sum(dim=-1)
        log_p = log_px_z + log_pz
        log_w = log_px_z + log_pz - log_q
        return log_w, log_p, log_q

    def loss(self, log_w, log_p=None, log_q=None, L=0, obj_f='elbo'):
        if L == 0:
            L = self.L
        if obj_f == 'elbo':
            elbo = log_w.sum()
            return - elbo
        elif obj_f == 'iwelbo':
            return - torch.sum(torch.logsumexp(log_p - log_q, dim=0) - np.log(L))
        elif obj_f == 'beta':
            beta_obj = log_w.sum()
            return - beta_obj
        elif obj_f == "miselbo":
            return - torch.logsumexp(log_w - np.log(L), dim=0).mean(dim=-1).sum()

    def backpropagate(self, x, z, mu, std, recon):
        log_w, log_p, log_q = self.get_log_w(x, z, mu, std, recon)

        # reset gradients
        self.phi_optim.zero_grad()

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= mu.size(0)  # divide by batch size

        if self.share_theta:
            loss.backward()
        else:
            self.theta_optim.zero_grad()
            loss.backward(retain_graph=True)
            self.theta_optim.step()
        self.phi_optim.step()
        return loss

    def get_entropy(self, z, mu, std):
        # (bs, z_dims)
        return 0.5 * self.z_dims * (np.log(2 * np.pi) + 1) + 0.5 * torch.sum(torch.log(std), dim=-1)

    @staticmethod
    def get_reconstruction_loss(x, recon):
        # (J, bs, x_dims)
        return torch.mean(torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1), dim=0)


class VanillaVAE(VAE):
    def __init__(self, decoder=None, L=1, lr=1e-3, share_theta=False, seed=0, x_dims=784, z_dims=40, device='cuda:0'):
        super().__init__(decoder=decoder, L=L, share_theta=share_theta, seed=seed,
                         x_dims=x_dims, z_dims=z_dims, device=device)
        self.model_name = f"VanillaVAE_L_{L}_seed_{seed}"
        self.phi = self.encoder.parameters()

        if not share_theta:
            self.theta = self.decoder.parameters()
            self.theta_optim = torch.optim.Adam(params=self.theta, lr=lr, weight_decay=0)

        self.phi_optim = torch.optim.Adam(params=self.phi, lr=lr, weight_decay=0)

        self.obj_f = 'elbo'


class BetaVAE(VAE):
    def __init__(self, decoder=None, L=1, lr=1e-3, share_theta=False, seed=0, x_dims=784,
                 z_dims=40, beta=1., device='cuda:0'):
        super().__init__(decoder=decoder, L=L, share_theta=share_theta, seed=seed,
                         x_dims=x_dims, z_dims=z_dims, device=device)
        self.model_name = f"BetaVAE_a_{beta}_seed_{seed}"
        self.phi = self.encoder.parameters()
        self.beta = beta

        if not share_theta:
            self.theta = self.decoder.parameters()
            self.theta_optim = torch.optim.Adam(params=self.theta, lr=lr, weight_decay=0)

        self.phi_optim = torch.optim.Adam(params=self.phi, lr=lr, weight_decay=0)

        self.obj_f = 'beta'

    def get_log_w(self, x, z, mu, std, recon):
        log_px_z = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)
        log_pz = Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
        log_q = Normal(mu, std).log_prob(z).sum(dim=-1)
        log_p = log_px_z + log_pz
        log_w = log_px_z + self.beta * (log_pz - log_q)
        return log_w, log_p, log_q


class VampVAE(VAE):
    def __init__(self, decoder=None, L=1, lr=1e-3, weight_decay=0, share_theta=False, seed=0, K=2,
                 alpha=1., beta=1., vamp_net=None, x_dims=784, z_dims=40, obj_f='beta'):
        super().__init__(decoder=decoder, L=L, share_theta=share_theta, seed=seed, x_dims=x_dims, z_dims=z_dims)

        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.model_name = f"VampVAE_{K}"
        self.obj_f = obj_f

        if vamp_net is not None:
            # parameters in vamp net are not trainable
            self.vamp_net = vamp_net
            self.K = self.vamp_net.K
            self.phi = self.encoder.parameters()
        else:
            # both vamp net and encoder net params are
            self.vamp_net = VampNet(encoder=self.encoder, n_dims=self.x_dims, K=self.K)
            self.phi = list(self.encoder.parameters()) + list(self.vamp_net.pseudo_generator.parameters())

        self.phi_optim = torch.optim.Adam(params=self.phi, lr=lr, weight_decay=weight_decay)

        if not share_theta:
            assert decoder is None
            self.theta = self.decoder.parameters()
            self.theta_optim = torch.optim.Adam(params=self.theta, lr=lr, weight_decay=weight_decay)

    def get_log_w(self, x, z, mu, std, recon, training=False):
        log_px_z = torch.sum(
            x * torch.log(recon + 1e-8) +
            (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)
        p_mu, p_std = self.vamp_net()
        log_pz = self.vamp_likelihood(z, p_mu, p_std, fast=True)
        log_p = log_px_z + log_pz
        log_q = Normal(mu, std).log_prob(z).sum(dim=-1)
        log_w = log_px_z + self.beta * (log_pz - log_q)

        return log_w, log_p, log_q

    def vamp_likelihood(self, z, p_mu, p_std, fast=True):
        if fast:
            shape_prior_params = (1, 1, self.K, self.z_dims)
            shape_prior_z_s = (z.size(0), z.size(1), 1, self.z_dims)
            Pz = Normal(p_mu.view(shape_prior_params), p_std.view(shape_prior_params))
            log_pz = torch.logsumexp(Pz.log_prob(z.view(shape_prior_z_s)).sum(dim=-1) - np.log(self.K), dim=-1)
        else:
            log_pz = torch.logsumexp(torch.stack([
                Normal(m, s).log_prob(z).sum(dim=-1) - np.log(self.K)
                for m, s in zip(p_mu, p_std)
            ]), dim=0)
        return log_pz

