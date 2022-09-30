from typing import Callable, Iterable, Optional

import numpy as np
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.nn as nn
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from prom_scvi import NegativeBinomial, ZeroInflatedNegativeBinomial
from prom_base_components import Encoder, DecoderSCVI
#from PriorNet import VampNet
from prom_scvi import Literal
from prom_scvi import auto_move_data



# Super class for VAEs. Use the VanillaVAE, AlphaVAE or ChiVAE classes instead
class VAE(nn.Module):
    def __init__(
            self,
            n_input: int,
            decoder=None,
            px_r = None,
            L=1,
            device='cuda:6',
            share_theta=False,
            seed=0,
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
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
            use_observed_lib_size: bool = True,
            library_log_means: Optional[np.ndarray] = None,
            library_log_vars: Optional[np.ndarray] = None,
            var_activation: Optional[Callable] = None,
    ):

        super(VAE, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        self.device = device
        self.share_theta = share_theta

        # number of importance samples
        self.L = L
        self.alpha = None
        self.beta = 1.

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        self.use_observed_lib_size = use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.dispersion == "gene":
            if px_r is not None:
                self.px_r = px_r.to(self.device)
            else:
                self.px_r = torch.nn.Parameter(torch.randn(n_input)).to(self.device)
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        ).to(self.device)
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution='ln',
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        ).to(self.device)
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov


        if decoder is not None:
            self.decoder = decoder.to(self.device)
        else:
            self.decoder = DecoderSCVI(
                n_input_decoder,
                n_input,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
            ).to(self.device)


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
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                library = Normal(ql_m, ql_v.sqrt()).sample()

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

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout) -> torch.Tensor:
        if self.gene_likelihood == "zinb":
            reconst_loss = (
                ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout, device=self.device
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss


    def get_log_w(self, x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout):
        log_px_z = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        log_pz = Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
        log_pl = Normal(torch.zeros_like(library), torch.ones_like(library)).log_prob(library).sum(dim=-1)

        log_qz = Normal(qz_m, qz_v).log_prob(z).sum(dim=-1)
        log_ql = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

        log_p = log_px_z + log_pz + log_pl
        log_q = log_qz + log_ql
        log_w = log_p - log_q
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

    def backpropagate(self, x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout):
        log_w, log_p, log_q = self.get_log_w(x, z, library, qz_m, qz_v, ql_m, ql_v, px_rate, px_r, px_dropout)

        # reset gradients
        self.phi_optim.zero_grad()

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= qz_m.size(0)  # divide by batch size

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


