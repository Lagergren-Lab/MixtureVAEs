import numpy as np
import os
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.nn as nn
from torch.distributions import Normal
from prom_base_components import Encoder, DecoderSCVI, EnsembleEncoders
#from PriorNet import VampNet
from prom_vae import VAE
from typing import Callable, Iterable, Optional
from prom_scvi import Literal
from prom_scvi import auto_move_data

class MISVAE(VAE):
    def __init__(self,
                 n_input: int,
                 S=2,
                 decoder=None,
                 model_name="model",
                 train_pi=False,
                 L=1,
                 device='cuda:6',
                 seed=0,
                 beta=1.,
                 lr=1e-3,
                 ais=False,
                 n_batch: int = 0,
                 n_labels: int = 0,
                 n_hidden: int = 128,
                 n_latent: int = 10,
                 n_layers: int = 1,
                 n_continuous_cov: int = 0,
                 n_cats_per_cov: Optional[Iterable[int]] = None,
                 dropout_rate: float = 0.1,
                 dispersion: str = "gene",
                 log_variational: bool = True,
                 gene_likelihood: str = "zinb",
                 latent_distribution: str = "normal",
                 encode_covariates: bool = False,
                 deeply_inject_covariates: bool = True,
                 use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
                 use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
                 use_observed_lib_size: bool = True,
                 library_log_means: Optional[np.ndarray] = None,
                 library_log_vars: Optional[np.ndarray] = None,
                 var_activation: Optional[Callable] = None
                 ):


        super().__init__(decoder=decoder,
                         L=L,
                         seed=seed,
                         device=device,
                         n_input = n_input,
                         n_labels = n_labels,
                         n_hidden = n_hidden,
                         n_latent = n_latent,
                         n_layers = n_layers,
                         n_continuous_cov = n_continuous_cov,
                         n_cats_per_cov = n_cats_per_cov,
                         dropout_rate = dropout_rate,
                         dispersion = dispersion,
                         log_variational = log_variational,
                         gene_likelihood = gene_likelihood,
                         latent_distribution = latent_distribution,
                         encode_covariates = encode_covariates,
                         deeply_inject_covariates = deeply_inject_covariates,
                         use_batch_norm = use_batch_norm,
                         use_layer_norm = use_layer_norm,
                         use_observed_lib_size = use_observed_lib_size,
                         library_log_means = library_log_means,
                         library_log_vars = library_log_vars,
                         var_activation = var_activation
                         )


        self.S = S
        self.device = device
        self.model_name = model_name
        if ais:
            self.encoder = None
        else:
            self.z_encoder = EnsembleEncoders(S=S,
                 n_input = n_input,
                 n_output = n_latent,
                 n_layers = n_layers,
                 n_hidden = n_hidden,
                 dropout_rate = dropout_rate,
                 var_activation = var_activation,
                 device=device,
                 gated=True).to(self.device)

            self.l_encoder = EnsembleEncoders(S=S,
                 n_input = n_input,
                 n_output = 1,
                 n_layers = 1,
                 n_hidden = n_hidden,
                 dropout_rate = dropout_rate,
                 var_activation = var_activation,
                 device=device,
                 gated=True).to(self.device)

            #self.phi_z = self.z_encoder.parameters()
            #self.phi_l = self.l_encoder.parameters()
            #self.theta = self.decoder.parameters()
            #self.optim = torch.optim.Adam(params=list(self.phi_z) + list(self.phi_l) + list(self.theta), lr=lr, weight_decay=0)
            self.optim = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=0)

        #self.model_name = f"MISVAE_a_{beta}_seed_{seed}_S_{S}"
        self.obj_f = 'miselbo'

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


    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, L=0):
        if L == 0:
            L = self.L
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        n_samples = L
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1).to(self.device)
        if self.log_variational:
            x_ = torch.log(1 + x_).to(self.device)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_.to(self.device)
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        qz_m, qz_v = self.z_encoder(encoder_input, *categorical_input)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v = self.l_encoder(
                encoder_input, *categorical_input
            )

        untran_z = self.sample(qz_m, torch.sqrt(qz_v), L)
        z = self.z_encoder.z_transformation(untran_z)

        if self.use_observed_lib_size:
            library = library.unsqueeze(1).expand((L, library.size(0), self.S,  library.size(1)) ).to(self.device)
        else:
            untran_lib = self.sample(ql_m, torch.sqrt(ql_v), L)
            library = self.l_encoder.z_transformation(untran_lib)

        #if n_samples > 1:
            #qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            #qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            #untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            #z = self.z_encoder.z_transformation(untran_z)
            #if self.use_observed_lib_size:
                #library = library.unsqueeze(0).expand(
                    #(n_samples, library.size(0), library.size(1))
                #)
            #else:
                #ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                #ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                #library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs



    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch


        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, decoder_input, library, batch_index, *categorical_input, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )


    def loss(self, log_w, log_p=None, log_q=None, L=0, obj_f='miselbo'):
        if L == 0:
            L = self.L
        if obj_f == 'elbo':
            elbo = log_w.sum()
            return - elbo
        elif obj_f == 'iwelbo':
            return - 1/self.S * torch.sum(torch.logsumexp(log_p - log_q - np.log(L), dim=0))
        elif obj_f == 'miselbo_beta':
            beta_obj = log_w.mean(dim=-1).sum()
            return - beta_obj
        elif obj_f == "miselbo":
            return - torch.sum(torch.mean(torch.logsumexp(log_p - log_q - np.log(L), dim=0), dim=-1))

    def backpropagate(self, x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout):
        log_w, log_p, log_q = self.get_log_w(x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout)

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= qz_m.size(0)  # divide by batch size
        loss.backward()

        # take step
        self.optim.step()

        # reset gradients
        self.optim.zero_grad()
        return loss


    def get_log_w(self, x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout):
        # z has dims L, bs, S, z_dims
        L, bs = z.size(0), z.size(1)
        #print(z.shape)
        #print(x.shape)

        x = x.view((1, bs, 1, x.size(-1)))
        #print(x.shape)
        log_px_z = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        #print('log_px_z')
        #print(log_px_z)
        # z has dims L, bs, S, z_dims
        log_Q = torch.zeros((z.size(0), z.size(1), self.S)).to(self.device)
        # mu has dims 1, bs, S, z_dims
        Q_mixt = Normal(qz_m, torch.sqrt(qz_v))

        log_pz = torch.zeros_like(log_Q)
        for s in range(self.S):
            # get z from component s and expand to fit Q_mixt dimensions
            z_s = z[..., s, :].view((z.size(0), z.size(1), 1, z.size(-1)))
            # compute likelihood of z_s according to the variational ensemble
            # log_Q_mixture_wrt_z_s = torch.logsumexp(Q_mixt.log_prob(z_s).sum(dim=-1) - np.log(self.S), dim=-1)
            log_Q_mixture_wrt_z_s = torch.logsumexp(Q_mixt.log_prob(z_s).sum(dim=-1) - np.log(self.S), dim=-1)
            log_Q[..., s] = log_Q_mixture_wrt_z_s
            #print('log_Q_mixture_wrt_z_s')
            #print(log_Q_mixture_wrt_z_s)
            log_pz[..., s] = self.compute_prior(z_s)
            #print('self.compute_prior(z_s)')
            #print(self.compute_prior(z_s))

        log_p = log_px_z + log_pz
        #print('log_p')
        #print(log_p)
        log_w = log_px_z + self.beta * (log_pz - log_Q)
        #print('log_w')
        #print(log_w[0:5])
        return log_w, log_p, log_Q

    def compute_prior(self, z):
        z = z.squeeze(-2)
        return Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)


    def forward(self, x, L=0, batch_index=None, y=None):
        if L == 0:
            L = self.L

        encoder_outputs = self.inference(x, batch_index, cont_covs=None, cat_covs=None, L=L)
        qz_m = encoder_outputs["qz_m"]
        qz_v = encoder_outputs["qz_v"]
        ql_m = encoder_outputs["ql_m"]
        ql_v = encoder_outputs["ql_v"]
        z = encoder_outputs["z"]
        library = encoder_outputs["library"]

        decoder_outputs = self.generative(z, library, batch_index, cont_covs=None, cat_covs=None, y=None,
                                          transform_batch=None,)

        px_scale = decoder_outputs["px_scale"]
        px_r = decoder_outputs["px_r"]
        px_rate = decoder_outputs["px_rate"]
        px_dropout = decoder_outputs["px_dropout"]

        return z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout



    def trainer(self, train_dataloader, val_dataloader, dir_ = './', n_epochs=200,
                verbose=True, L=50, warmup=None, N=100, val_obj_f="miselbo"):
        if warmup == "kl_warmup":
            vae.beta = 0
        self.train()
        train_loss_avg = np.zeros(n_epochs)
        eval_loss_avg = []
        best_nll = 1e10
        best_epoch = 0
        for epoch in range(n_epochs):
            num_batches = 0
            if warmup == "kl_warmup":
                self.beta = np.minimum(1 / (N - 1) * epoch, 1.)
            for x in train_dataloader:
                #x = x.to(self.device).float().view((-1, self.x_dims))
                #x = torch.bernoulli(x)
                # forward
                z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = self.forward(x)
                # backward
                loss = self.backpropagate(x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout)
                train_loss_avg[epoch] += loss.item()
                num_batches += 1
            test_nll = self.evaluate(val_dataloader)
            train_loss_avg[epoch] /= num_batches
            eval_loss_avg.append(test_nll)
            if test_nll < best_nll:
                path = os.path.join(dir_, self.model_name)
                torch.save(self.state_dict(), path)
                best_nll = test_nll
                best_epoch = epoch
            elif (epoch - best_epoch) >= 100:
                return train_loss_avg, eval_loss_avg
            if verbose and epoch % 50 == 0:
                print("Epoch: ", epoch)
                print(f"Test NLL: ", test_nll, f" ({round(best_nll, 2)}; {best_epoch})")
                if warmup == "kl_warmup":
                    print("Beta: ", round(vae.beta, 2))
        return train_loss_avg, eval_loss_avg

    def evaluate(self, dataloader, L=0, obj_f='miselbo'):
        if L == 0:
            L = self.L
        elbo = 0
        num_batches = 0
        for x in dataloader:
            #x = x.to(self.device).float().view((-1, vae.x_dims))
            with torch.no_grad():
                z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = self(x, L)
                log_w, log_p, log_q = self.get_log_w(x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout)
                loss = self.loss(log_w, log_p, log_q, L, obj_f=obj_f)
                elbo += loss.item()
                num_batches += len(x)
        avg_elbo = elbo / num_batches
        return avg_elbo







