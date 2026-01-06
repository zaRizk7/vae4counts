import pyro
import torch
import torch.nn as nn
from pyro.distributions import Normal
from pyro.poutine import scale

from ..activations import make_feature_scaler
from ..layers.variational import VariationalMLP
from .mixture import GaussianMixture


class VAE(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_latents: int,
        output_dist: str = "gaussian",
        feature_scaler: str = "identity",
        num_hiddens_encoder: list[int] | None = None,
        num_hiddens_decoder: list[int] | None = None,
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        scaler_kwargs: dict = {},
        obs_name: str = "obs",
        latent_name: str = "latent",
        module_name: str = "vae",
    ):
        super().__init__()
        self.feature_scaler = make_feature_scaler(feature_scaler, **scaler_kwargs)
        self.encoder = VariationalMLP(
            "gaussian",
            num_features,
            num_latents,
            num_hiddens_encoder,
            encoder_kwargs.get("mlp", {}),
            encoder_kwargs.get("variational", {}),
        )
        self.decoder = VariationalMLP(
            output_dist,
            num_latents,
            num_features,
            num_hiddens_decoder,
            decoder_kwargs.get("mlp", {}),
            variational_kwargs={
                "constant_scale": True,
                **decoder_kwargs.get("variational", {}),
            },
        )

        self.latent_name = latent_name
        self.obs_name = obs_name
        self.module_name = module_name
        self.mean_prior = nn.Buffer(torch.zeros(num_latents))
        self.std_prior = nn.Buffer(torch.ones(num_latents))

    @property
    def latent_prior(self):
        return Normal(self.mean_prior, self.std_prior).to_event(1)

    def model(self, obs: torch.Tensor, kl_scale: float = 1.0):
        pyro.module(self.module_name, self)
        with pyro.plate("N", len(obs)):
            with scale(scale=kl_scale):
                latent = pyro.sample(self.latent_name, self.latent_prior)
            return pyro.sample(self.obs_name, self.decoder(latent), obs=obs)

    def guide(self, obs: torch.Tensor, kl_scale: float = 1.0):
        obs = self.feature_scaler(obs)
        pyro.module(self.module_name, self)
        with pyro.plate("N", len(obs)), scale(scale=kl_scale):
            return pyro.sample(self.latent_name, self.encoder(obs))

    def forward(self, obs: torch.Tensor | None = None, decode: bool = False):
        if obs is not None:
            latent_dist = self.latent_prior
        else:
            obs = self.feature_scaler(obs)
            latent_dist = self.encoder(obs)

        if not decode:
            return latent_dist

        if latent_dist.has_rsample:
            latent = latent_dist.rsample()
        else:
            latent = latent_dist.sample()

        return self.decoder(latent)


class GMVAE(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_latents: int,
        num_components: int,
        output_dist: str = "gaussian",
        feature_scaler: str = "identity",
        num_hiddens_mixing_encoder: list[int] | None = None,
        num_hiddens_comp_encoder: list[int] | None = None,
        num_hiddens_decoder: list[int] | None = None,
        mixing_encoder_kwargs: dict = {},
        comp_encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        gmm_kwargs: dict = {},
        scaler_kwargs: dict = {},
        obs_name: str = "obs",
        comp_name: str = "comp",
        latent_name: str = "latent",
        module_name: str = "gmvae",
    ):
        super().__init__()
        self.feature_scaler = make_feature_scaler(feature_scaler, **scaler_kwargs)
        self.mixing_encoder = VariationalMLP(
            "categorical",
            num_features,
            num_components,
            num_hiddens_mixing_encoder,
            mixing_encoder_kwargs.get("mlp", {}),
            mixing_encoder_kwargs.get("variational", {}),
        )
        self.comp_encoder = VariationalMLP(
            "gaussian_mixture",
            num_features,
            num_latents,
            num_hiddens_comp_encoder,
            comp_encoder_kwargs.get("mlp", {}),
            variational_kwargs={
                "num_components": num_components,
                **comp_encoder_kwargs.get("variational", {}),
            },
        )
        self.decoder = VariationalMLP(
            output_dist,
            num_latents,
            num_features,
            num_hiddens_decoder,
            decoder_kwargs.get("mlp", {}),
            variational_kwargs={
                "constant_scale": True,
                **decoder_kwargs.get("variational", {}),
            },
        )

        self.comp_name = comp_name
        self.obs_name = obs_name
        self.module_name = module_name
        self.gmm = GaussianMixture(
            num_latents,
            num_components,
            comp_name=comp_name,
            obs_name=latent_name,
            **gmm_kwargs,
        )

    def model(
        self,
        obs: torch.Tensor,
        kl_scale_mixing: float = 1.0,
        kl_scale_comp: float = 1.0,
    ):
        pyro.module(self.module_name, self)
        with pyro.plate("N", len(obs)):
            latent = self.gmm(
                kl_scale_mixing=kl_scale_mixing, kl_scale_comp=kl_scale_comp
            )
            return pyro.sample(self.obs_name, self.decoder(latent), obs=obs)

    def guide(
        self,
        obs: torch.Tensor,
        kl_scale_mixing: float = 1.0,
        kl_scale_comp: float = 1.0,
    ):
        obs = self.feature_scaler(obs)
        pyro.module(self.module_name, self)
        with pyro.plate("N", len(obs)):
            with scale(scale=kl_scale_mixing):
                comp = pyro.sample(
                    self.gmm.comp_name, self.mixing_encoder(obs), **self.gmm.infer_cfg
                )
            with scale(scale=kl_scale_comp):
                return pyro.sample(self.gmm.obs_name, self.comp_encoder(obs, comp))

    def forward(self, obs: torch.Tensor | None = None, decode: bool = False):
        if obs is not None:
            obs = self.feature_scaler(obs)
            mixing_dist = self.mixing_encoder(obs)
            latent_dist = self.comp_encoder(obs, mixing_dist.probs)
        else:
            latent_dist = self.gmm.distribution

        if not decode:
            return mixing_dist, latent_dist

        if latent_dist.has_rsample:
            latent = latent_dist.rsample()
        else:
            latent = latent_dist.sample()

        return self.decoder(latent)
