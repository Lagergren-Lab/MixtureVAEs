import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from data.load_data import load_bimodal, load_shell
import matplotlib.pyplot as plt
# from models.VAEIAF import NIAFEncoder


def sample(mu, std, K, bs):
    latent_dims = z_dims
    eps = torch.randn((K, bs, S, latent_dims)).to(device)
    mu = mu.unsqueeze(0).expand((K, bs, S, latent_dims))
    std = std.unsqueeze(0).expand((K, bs, S, latent_dims))
    sample = mu + eps * std
    return sample.float()


def get_log_p(z):
    log_pz = torch.logsumexp(torch.stack([
        Normal(torch.zeros_like(z) + m, 0.5 * torch.ones_like(z)).log_prob(z).sum(dim=-1) + np.log(pi)
        for m, pi in zip([-1, 1], [0.2, 0.8])
    ]), dim=0)
    return log_pz


def get_log_p_np(z):
    return get_log_p(torch.tensor(z))


def encode(self, z):
    log_det = torch.zeros((z.size(0), z.size(1)), device=self.device)

    for flow in self.flows:
        sm = flow.forward(z)
        s, m = sm.split(self.dim, dim=-1)

        std = torch.sigmoid(s)
        z = std*z + (1-std) * m
        log_det += -torch.sum(torch.log(std), dim=-1).squeeze(-1)

    return z, log_det


def scatter_plot_weighted(z, w):
    z = z.view((z.size(0) * z.size(1), z.size(2), z.size(-1))).cpu().detach().numpy()
    for s in range(S):
        plt.scatter(z[..., s, 0], z[..., s, 1], c=w[..., s].cpu().detach())
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('off')
    plt.xlabel('$z_1$')
    plt.xlabel('$z_2$')
    # plt.title(f'$S={S}; L={L}$')
    # plt.title(f'S=1; IAF w. $T=30$')
    plt.show()


def scatter_plot(z):
    z = z.view((z.size(0) * z.size(1), z.size(2), z.size(-1))).cpu().detach().numpy()
    for s in range(S):
        plt.scatter(z[..., s, 0], z[..., s, 1])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('off')
    plt.xlabel('$z_1$')
    plt.xlabel('$z_2$')
    # plt.title(f'$S={S}; L={L}$')
    # plt.title(f'S=1; IAF w. $T=30$')
    plt.show()


seed = 1
torch.manual_seed(seed)
S = 1
z_dims = 2
L = 1
bs = 100
device = 'cuda:0'

z = np.random.uniform(-2, 2, (10000, 2))
w = np.exp(get_log_p_np(z))
mask = w > 0.1
plt.scatter(z[mask][:, 0], z[mask][:, 1])
plt.axis('off')
plt.xlabel('$z_1$')
plt.xlabel('$z_2$')
# plt.title(f'$p(z)$')
plt.show()


# iaf = NIAFEncoder(device='cuda:0', dim=2, nh=10, T=30, seed=seed).to('cuda:0')

mu = torch.tensor(np.random.uniform(-0, 0, (S, z_dims)), device=device).requires_grad_()
log_var = (torch.zeros((S, z_dims)).to(device)).requires_grad_()
# optim = torch.optim.Adam(params=[mu] + [log_var] + list(iaf.parameters()), lr=.1, weight_decay=0)
optim = torch.optim.Adam(params=[mu] + [log_var], lr=.1, weight_decay=0)
for i in range(5000):
    optim.zero_grad()

    std = torch.exp(0.5 * log_var)
    z = sample(mu, std, L, bs)

    log_Q = torch.zeros((z.size(0), z.size(1), S, S)).to(device)
    Q_mixt = Normal(mu, std)
    # zT, log_det = encode(iaf, z)
    for s in range(S):
        z_k = z[..., s, :].view((z.size(0), z.size(1), 1, z.size(-1)))
        log_Q[..., s, :] = Q_mixt.log_prob(z_k).sum(dim=-1)  # + log_det.unsqueeze(-1)
    log_pz = get_log_p(z)
    # idx = torch.arange(S)
    # log_w = log_pz - log_Q[:, :, idx, idx]
    log_w = log_pz - torch.logsumexp(log_Q - np.log(S), dim=-1)
    miselbo = -torch.mean(torch.logsumexp(log_w, dim=0) - np.log(L), dim=-1).mean()
    # compute losses
    miselbo.backward()
    optim.step()

    if i % 1000 == 0:
        with torch.no_grad():
            z = sample(mu, std, 30, bs)
            # z, log_det = encode(iaf, z)
            log_pz = get_log_p(z)
        scatter_plot_weighted(z, torch.exp(log_pz))
        print(miselbo.detach().item())


std = torch.exp(0.5 * log_var)
z = sample(mu, std, 30, bs)
# z, _ = encode(iaf, z)
w = get_log_p(z)
scatter_plot(z)
scatter_plot_weighted(z, torch.exp(w))







