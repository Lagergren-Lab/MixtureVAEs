import numpy as np
import torch
from torch.distributions import Normal
from models.misvae import MISVAECNN
from models.VAEIAF import EnsembleIAFs


class NFMISVAE(MISVAECNN):
    def __init__(self, S=2, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3, T=2):
        super().__init__(L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device, S=S)
        self.S = S
        self.device = device

        self.flow = EnsembleIAFs(device, dim=z_dims, nh=320, T=T, S=S, seed=seed)
        self.phi = list(self.encoder.parameters()) + list(self.flow.parameters())
        self.theta = self.decoder.parameters()
        self.optim = torch.optim.Adam(params=self.phi + list(self.theta), lr=lr, weight_decay=0)

        self.model_name = f"NFMISVAE_a_{beta}_seed_{seed}_S_{S}"
        self.obj_f = 'miselbo_beta'

        # number of importance samples
        self.L = L
        self.beta = beta

    def forward(self, x, L=0):
        if L == 0:
            L = self.L
        mu, std = self.encoder(x)
        z = self.sample(mu, std, L)
        zT = torch.zeros((L, z.size(1), self.S, self.S, self.z_dims), device=self.device)
        log_Q = torch.zeros((z.size(0), z.size(1), self.S, self.S)).to(self.device)
        for s in range(self.S):
            zT[:, :, s, :, :], log_Q[:, :, s, :] = self.flow(z[..., s, :])
        idx = torch.arange(mu.size(1))
        reconstruction = self.decoder(x, zT[:, :, idx, idx, :])
        return z, mu, std, reconstruction, zT, log_Q

    def get_log_w(self, x, z0, mu, std, recon, zT, log_det):
        # z has dims L, bs, S, z_dims
        L, bs = z0.size(0), x.size(0)
        x = x.view((1, bs, 1, 784))

        log_px_z = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)

        # mu has dims 1, bs, S, z_dims
        Q_mixt = Normal(mu, std)
        # z has dims L, bs, S, z_dims
        log_Q = torch.zeros((z0.size(0), z0.size(1), self.S)).to(self.device)

        log_pz = torch.zeros_like(log_Q)
        for s in range(self.S):
            # get z from component s and expand to fit Q_mixt dimensions
            z0_s = z0[..., s, :].view((z0.size(0), z0.size(1), 1, z0.size(-1)))
            zT_s = zT[..., s, s, :].view((z0.size(0), z0.size(1), 1, z0.size(-1)))
            # compute likelihood of z_s according to the variational ensemble
            log_Q_mixture_wrt_z_s = torch.logsumexp(
                Q_mixt.log_prob(z0_s).sum(dim=-1) + log_det[..., s, :] - np.log(self.S),
                dim=-1)
            log_Q[..., s] = log_Q_mixture_wrt_z_s
            log_pz[..., s] = self.compute_prior(zT_s)

        log_p = log_px_z + log_pz
        log_w = log_px_z + self.beta * (log_pz - log_Q)
        return log_w, log_p, log_Q

    def backpropagate(self, x):
        z, mu, std, reconstruction, zT, log_Q = self.forward(x)
        log_w, log_p, log_q = self.get_log_w(x, z, mu, std, reconstruction, zT, log_Q)

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= mu.size(0)  # divide by batch size
        loss.backward()

        # take step
        self.optim.step()

        # reset gradients
        self.optim.zero_grad()
        return loss

