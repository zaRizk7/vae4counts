import pyro
import torch
import torch.nn as nn
from pyro.distributions import Normal
from pyro.poutine import scale
from torch.distributions import kl_divergence

from ..activations import make_feature_scaler
from ..layers.probabilistic import ProbabilisticMLP
from .mixture import GaussianMixture

__all__ = ["VAE", "GMVAE", "CountVAE", "CountGMVAE"]


class CountModelMixin:
    """A mixin class specific for modeling count data. Mainly used with
    RNA-seq data that tend to have large library size (sum of counts in
    a sample). It builds a log-Gaussian prior from count data's library size
    (total counts per sample). This type of prior was also used in scVI, but this
    implementation is a more simplified version without batch effect adjustments.
    """

    def fit_freq_scale_prior(self, obs: torch.Tensor):
        """Fits a univariate gaussian from the log-scale of the observation's
        library size/sum over all counts per sample. Used typically to build
        Empirical Bayes prior for scaling gene expressions.

        Args:
            obs (torch.Tensor): All observations (full dataset) used to fit the prior.
        """
        prior = GaussianMixture.fit_from_data(obs.sum(-1), log=True)
        self.freq_scale_mean_prior = nn.Buffer(prior.loc[0])
        self.freq_scale_std_prior = nn.Buffer(prior.scale[0])

        return self

    @property
    def freq_scale_prior(self):
        """Distribution from the fitted prior."""
        loc, scale = self.freq_scale_mean_prior, self.freq_scale_std_prior
        return Normal(loc, scale).to_event(1)


class VAE(nn.Module):
    """Variational autoencoder (VAE) with standard Gaussian prior.

    Args:
        num_features (int): Number of input features.
        num_latents (int): Number of latent features.
        output_dist (str): Distribution to model the features. Either
            "categorical", "gaussian", "gaussian_mixture", "poisson", "negative_binomial", "zi_poisson" or "zi_nb".
            Default: "gaussian"
        feature_scaler (str): Feature scaler for the input before encoding. Either
            "identity", "standard", "cps", "log1p", or "log1pcps". Default: "identity"
        num_hiddens_latent_encoder (list[int] | None): List of number of hidden sizes
            for latent encoder. Default: None
        num_hiddens_decoder (list[int] | None): List of number of hidden sizes for decoder. Default: None
        encoder_kwargs (dict): Additional params for initializing the encoder. Default: {}
        decoder_kwargs (dict): Additional params for initializing the decoder. Default: {}
        scaler_kwargs (dict): Additional params for initializing the feature scaler. Default: {}
        obs_name (str): Variable name for the observed variable. Default: "obs"
        latent_name (str): Variable name for the Gaussian latent variable. Default: "latent"
        module_name (str): A name to register the module to Pyro. Normally used if this module
            only used for another module extensions. Default: "vae"
    """

    def __init__(
        self,
        num_features: int,
        num_latents: int,
        output_dist: str = "gaussian",
        feature_scaler: str = "identity",
        num_hiddens_latent_encoder: list[int] | None = None,
        num_hiddens_decoder: list[int] | None = None,
        latent_encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        scaler_kwargs: dict = {},
        obs_name: str = "obs",
        latent_name: str = "latent",
        module_name: str = "vae",
    ):
        super().__init__()
        self.feature_scaler = make_feature_scaler(feature_scaler, **scaler_kwargs)
        self.latent_encoder = ProbabilisticMLP(
            "gaussian",
            num_features,
            num_latents,
            num_hiddens_latent_encoder,
            latent_encoder_kwargs.get("mlp", {}),
            latent_encoder_kwargs.get("probabilistic", {}),
        )

        self.decoder = ProbabilisticMLP(
            output_dist,
            num_latents,
            num_features,
            num_hiddens_decoder,
            decoder_kwargs.get("mlp", {}),
            {
                "constant_scale": True,
                **decoder_kwargs.get("probabilistic", {}),
            },
        )

        self.latent_name = latent_name
        self.obs_name = obs_name
        self.module_name = module_name
        self.mean_prior = nn.Buffer(torch.zeros(num_latents))
        self.std_prior = nn.Buffer(torch.ones(num_latents))

    @property
    def latent_prior(self):
        """Standard Gaussian `N(0, I)` prior."""
        return Normal(self.mean_prior, self.std_prior).to_event(1)

    def model(
        self, obs: torch.Tensor, rec_scale: float = 1.0, latent_scale: float = 1.0
    ):
        """Sample from VAE's generative distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            latent_scale (float): Multiplicative factor for the latent term. Default: 1.0
        """
        pyro.module(self.module_name, self)

        with pyro.plate("N", len(obs)):
            with scale(scale=latent_scale):
                latent = pyro.sample(self.latent_name, self.latent_prior)

            with scale(scale=rec_scale):
                return pyro.sample(self.obs_name, self.decoder(latent), obs=obs)

    def guide(
        self, obs: torch.Tensor, rec_scale: float = 1.0, latent_scale: float = 1.0
    ):
        """Sample from VAE's variational distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            latent_scale (float): Multiplicative factor for the latent term. Default: 1.0
        """
        pyro.module(self.module_name, self)
        obs = self.feature_scaler(obs)

        with pyro.plate("N", len(obs)), scale(scale=latent_scale):
            return pyro.sample(self.latent_name, self.latent_encoder(obs))

    def forward(self, obs: torch.Tensor | None = None, decode: bool = False):
        if obs is None:
            latent_dist = self.latent_prior
        else:
            obs = self.feature_scaler(obs)
            latent_dist = self.latent_encoder(obs)

        if not decode:
            return latent_dist

        if latent_dist.has_rsample:
            latent = latent_dist.rsample()
        else:
            latent = latent_dist.sample()

        return self.decoder(latent)


class GMVAE(nn.Module):
    """Variational autoencoder (VAE) with Mixture of Gaussians prior.

    Args:
        num_features (int): Number of input features.
        num_latents (int): Number of latent features.
        num_components (int): Number of components for the mixture.
        output_dist (str): Distribution to model the features. Either
            "categorical", "gaussian", "gaussian_mixture", "poisson", "negative_binomial", "zi_poisson" or "zi_nb".
            Default: "gaussian"
        feature_scaler (str): Feature scaler for the input before encoding. Either
            "identity", "standard", "cps", "log1p", or "log1pcps". Default: "identity"
        num_hiddens_mixing_encoder (list[int] | None): List of number of hidden sizes for mixing weight encoder.
            Default: None
        num_hiddens_comp_encoder (list[int] | None): List of number of hidden sizes for component encoder.
            Default: None
        num_hiddens_decoder (list[int] | None): List of number of hidden sizes for decoder. Default: None
        mixing_encoder_kwargs (dict): Additional params for initializing the mixing weight encoder. Default: {}
        comp_encoder_kwargs (dict): Additional params for initializing the component weight encoder. Default: {}
        decoder_kwargs (dict): Additional params for initializing the decoder. Default: {}
        gmm_kwargs (dict): Additional params for initializing the mixture of Gaussians. Default: {}
        scaler_kwargs (dict): Additional params for initializing the feature scaler. Default: {}
        obs_name (str): Variable name for the observed variable. Default: "obs"
        comp_name (str): Variable name for the component discrete latent variable. Default: "comp"
        latent_name (str): Variable name for the Gaussian component latent variable. Default: "latent"
        module_name (str): A name to register the module to Pyro. Normally used if this module
            only used for another module extensions. Default: "gmvae"
    """

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
        self.mixing_encoder = ProbabilisticMLP(
            "categorical",
            num_features,
            num_components,
            num_hiddens_mixing_encoder,
            mixing_encoder_kwargs.get("mlp", {}),
            mixing_encoder_kwargs.get("probabilistic", {}),
        )
        self.comp_encoder = ProbabilisticMLP(
            "gaussian_mixture",
            num_features,
            num_latents,
            num_hiddens_comp_encoder,
            comp_encoder_kwargs.get("mlp", {}),
            {
                "num_components": num_components,
                **comp_encoder_kwargs.get("probabilistic", {}),
            },
        )
        self.decoder = ProbabilisticMLP(
            output_dist,
            num_latents,
            num_features,
            num_hiddens_decoder,
            decoder_kwargs.get("mlp", {}),
            {
                "constant_scale": True,
                **decoder_kwargs.get("probabilistic", {}),
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
        rec_scale: float = 1.0,
        mixing_scale: float = 1.0,
        comp_scale: float = 1.0,
        analytical_kl: bool = False,
    ):
        """Sample from VAE's generative distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            mixing_scale (float): Multiplicative factor for mixing distribution's KL
                regularization term. Default: 1.0
            comp_scale (float): Multiplicative factor for component distribution's KL
                regularization term. Default: 1.0
            analytical_kl (bool): Applies closed form KL formulation of GMVAE. Default: False
        """
        pyro.module(self.module_name, self)

        with pyro.plate("N", len(obs)):
            latent = self.gmm(mixing_scale=mixing_scale, comp_scale=comp_scale)
            with scale(scale=rec_scale):
                return pyro.sample(self.obs_name, self.decoder(latent), obs=obs)

    def guide(
        self,
        obs: torch.Tensor,
        rec_scale: float = 1.0,
        mixing_scale: float = 1.0,
        comp_scale: float = 1.0,
        analytical_kl: bool = False,
    ):
        """Sample from VAE's variational distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            mixing_scale (float): Multiplicative factor for mixing distribution's KL
                regularization term. Default: 1.0
            comp_scale (float): Multiplicative factor for component distribution's KL
                regularization term. Default: 1.0
            analytical_kl (bool): Applies closed form KL formulation of GMVAE. Default: False
        """
        pyro.module(self.module_name, self)
        obs = self.feature_scaler(obs)

        with pyro.plate("N", len(obs)):
            with scale(scale=mixing_scale):
                mixing_post = self.mixing_encoder(obs)
                comp = pyro.sample(
                    self.gmm.comp_name, mixing_post, **self.gmm.infer_cfg
                )
                if analytical_kl:
                    # Cancels the ratio to use analytical KL
                    mixing_prior = self.gmm.mixing_dist
                    ratio = mixing_prior.log_prob(comp) - mixing_post.log_prob(comp)
                    kl_mixing = kl_divergence(mixing_post, mixing_prior)
                    pyro.factor("ratio_mixing", ratio, has_rsample=False)
                    pyro.factor("kl_mixing", kl_mixing, has_rsample=False)

            with scale(scale=comp_scale):
                comp_post = self.comp_encoder(obs, comp)
                latent = pyro.sample(self.gmm.obs_name, comp_post)
                if analytical_kl:
                    # Cancels the ratio to use analytical KL
                    comp_prior = self.gmm.comp_dist(comp)
                    ratio = comp_prior.log_prob(latent) - comp_post.log_prob(latent)
                    kl_latent = kl_divergence(comp_post, comp_prior)
                    pyro.factor("ratio_comp", ratio, has_rsample=True)
                    pyro.factor("kl_latent", kl_latent, has_rsample=True)

        return latent

    def forward(self, obs: torch.Tensor | None = None, decode: bool = False):
        if obs is None:
            latent_dist = self.gmm.distribution
        else:
            obs = self.feature_scaler(obs)
            mixing_dist = self.mixing_encoder(obs)
            latent_dist = self.comp_encoder(obs, mixing_dist.probs)

        if not decode:
            return mixing_dist, latent_dist

        if latent_dist.has_rsample:
            latent = latent_dist.rsample()
        else:
            latent = latent_dist.sample()

        return self.decoder(latent)


class CountVAE(VAE, CountModelMixin):
    """Variational autoencoder (VAE) with standard Gaussian prior and
    frequency scaling prior for decoding counts.

    Args:
        num_features (int): Number of input features.
        num_latents (int): Number of latent features.
        output_dist (str): Distribution to model the features. Either
            "poisson", "negative_binomial", "zi_poisson" or "zi_nb".
            Default: "poisson"
        feature_scaler (str): Feature scaler for the input before encoding. Either
            "identity", "standard", "cps", "log1p", or "log1pcps". Default: "identity"
        num_hiddens_latent_encoder (list[int] | None): List of number of hidden sizes
            for latent encoder. Default: None
        num_hiddens_freq_scale_encoder (list[int] | None): List of number of hidden sizes
            for the frequency scale encoder. Default: None
        num_hiddens_decoder (list[int] | None): List of number of hidden sizes for decoder. Default: None
        latent_encoder_kwargs (dict): Additional params for initializing the latent encoder. Default: {}
        freq_scale_encode_encoder_kwargs (dict): Additional params for initializing the
            frequency scaler encoder. Default: {}
        decoder_kwargs (dict): Additional params for initializing the decoder. Default: {}
        scaler_kwargs (dict): Additional params for initializing the feature scaler. Default: {}
        obs_name (str): Variable name for the observed variable. Default: "obs"
        latent_name (str): Variable name for the Gaussian latent variable. Default: "latent"
        module_name (str): A name to register the module to Pyro. Normally used if this module
            only used for another module extensions. Default: "count_vae"
    """

    def __init__(
        self,
        num_features: int,
        num_latents: int,
        output_dist: str = "poisson",
        feature_scaler: str = "identity",
        num_hiddens_latent_encoder: list[int] | None = None,
        num_hiddens_freq_scale_encoder: list[int] | None = None,
        num_hiddens_decoder: list[int] | None = None,
        latent_encoder_kwargs: dict = {},
        freq_scale_encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        scaler_kwargs: dict = {},
        obs_name: str = "obs",
        latent_name: str = "latent",
        freq_scale_name: str = "freq_scale",
        module_name: str = "count_vae",
    ):
        if "gaussian" in output_dist:
            raise ValueError("'gaussian' is not supported for CountVAE.")

        super().__init__(
            num_features,
            num_latents,
            output_dist,
            feature_scaler,
            num_hiddens_latent_encoder,
            num_hiddens_decoder,
            latent_encoder_kwargs,
            decoder_kwargs,
            scaler_kwargs,
            obs_name,
            latent_name,
            module_name,
        )
        self.freq_scale_encoder = ProbabilisticMLP(
            "gaussian",
            num_features,
            1,
            num_hiddens_freq_scale_encoder,
            freq_scale_encoder_kwargs.get("mlp", {}),
            freq_scale_encoder_kwargs.get("probabilistic", {}),
        )
        self.freq_scale_name = freq_scale_name

    def model(
        self,
        obs: torch.Tensor,
        rec_scale: float = 1.0,
        latent_scale: float = 1.0,
        freq_scale: float = 1.0,
    ):
        """Sample from CountVAE's generative distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            batch (torch.Tensor): One-hot encoded batch labels. Will be marginalized if None. Default: None
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            latent_scale (float): Multiplicative factor for the latent term. Default: 1.0
            freq_scale (float): Multiplicative factor for the frequency scaling term. Default: 1.0
        """
        if not hasattr(self, "freq_scale_prior"):
            raise ValueError(
                "Please call 'fit_from_data' first to estimate the frequency scaling prior."
            )
        pyro.module(self.module_name, self)

        with pyro.plate("N", len(obs)):
            with scale(scale=latent_scale):
                latent = pyro.sample(self.latent_name, self.latent_prior)

            with scale(scale=freq_scale):
                freq = pyro.sample(self.freq_scale_name, self.freq_scale_prior)

            with scale(scale=rec_scale):
                return pyro.sample(self.obs_name, self.decoder(latent, freq), obs=obs)

    def guide(
        self,
        obs: torch.Tensor,
        rec_scale: float = 1.0,
        latent_scale: float = 1.0,
        freq_scale: float = 1.0,
    ):
        """Sample from CountVAE's variational distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            batch (torch.Tensor): One-hot encoded batch labels. Will be marginalized if None. Default: None
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            latent_scale (float): Multiplicative factor for the latent term. Default: 1.0
            freq_scale (float): Multiplicative factor for the frequency scaling term. Default: 1.0
        """
        pyro.module(self.module_name, self)
        obs = self.feature_scaler(obs)

        with pyro.plate("N", len(obs)):
            with scale(scale=latent_scale):
                latent = pyro.sample(self.latent_name, self.latent_encoder(obs))

            with scale(scale=freq_scale):
                freq = pyro.sample(self.freq_scale_name, self.freq_scale_encoder(obs))

        return latent, freq

    def forward(self, obs: torch.Tensor | None = None, decode: bool = False):
        if obs is None:
            latent_dist = self.latent_prior
            freq_scale_dist = self.freq_scale_prior
        else:
            obs = self.feature_scaler(obs)
            latent_dist = self.latent_encoder(obs)
            freq_scale_dist = self.freq_scale_encoder(obs)

        if not decode:
            return latent_dist, freq_scale_dist

        if latent_dist.has_rsample:
            latent = latent_dist.rsample()
        else:
            latent = latent_dist.sample()

        if freq_scale_dist.has_rsample:
            freq_scale = freq_scale_dist.rsample()
        else:
            freq_scale = freq_scale_dist.sample()

        return self.decoder(latent, freq_scale)


class CountGMVAE(GMVAE, CountModelMixin):
    """Variational autoencoder (VAE) with Mixture of Gaussians prior and
    frequency scaling prior for decoding counts.

    Args:
        num_features (int): Number of input features.
        num_latents (int): Number of latent features.
        output_dist (str): Distribution to model the features. Either
            "poisson", "negative_binomial", "zi_poisson" or "zi_nb".
            Default: "poisson"
        feature_scaler (str): Feature scaler for the input before encoding. Either
            "identity", "standard", "cps", "log1p", or "log1pcps". Default: "identity"
        num_hiddens_mixing_encoder (list[int] | None): List of number of hidden sizes
            for mixing encoder. Default: None
        num_hiddens_comp_encoder (list[int] | None): List of number of hidden sizes
            for component encoder. Default: None
        num_hiddens_freq_scale_encoder (list[int] | None): List of number of hidden sizes
            for the frequency scale encoder. Default: None
        num_hiddens_decoder (list[int] | None): List of number of hidden sizes for decoder. Default: None
        mixing_encoder_kwargs (dict): Additional params for initializing the mixing weights encoder. Default: {}
        comp_encoder_kwargs (dict): Additional params for initializing the component encoder. Default: {}
        freq_scale_encode_encoder_kwargs (dict): Additional params for initializing the
            frequency scaler encoder. Default: {}
        decoder_kwargs (dict): Additional params for initializing the decoder. Default: {}
        gmm_kwargs (dict): Additional params for initializing the mixture of Gaussians. Default: {}
        scaler_kwargs (dict): Additional params for initializing the feature scaler. Default: {}
        obs_name (str): Variable name for the observed variable. Default: "obs"
        latent_name (str): Variable name for the Gaussian latent variable. Default: "latent"
        module_name (str): A name to register the module to Pyro. Normally used if this module
            only used for another module extensions. Default: "count_vae"
    """

    def __init__(
        self,
        num_features: int,
        num_latents: int,
        num_components: int,
        output_dist: str = "poisson",
        feature_scaler: str = "identity",
        num_hiddens_mixing_encoder: list[int] | None = None,
        num_hiddens_comp_encoder: list[int] | None = None,
        num_hiddens_freq_scale_encoder: list[int] | None = None,
        num_hiddens_decoder: list[int] | None = None,
        mixing_encoder_kwargs: dict = {},
        comp_encoder_kwargs: dict = {},
        freq_scale_encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        gmm_kwargs: dict = {},
        scaler_kwargs: dict = {},
        obs_name: str = "obs",
        comp_name: str = "comp",
        latent_name: str = "latent",
        freq_scale_name: str = "freq_scale",
        module_name: str = "count_gmvae",
    ):
        super().__init__(
            num_features,
            num_latents,
            num_components,
            output_dist,
            feature_scaler,
            num_hiddens_mixing_encoder,
            num_hiddens_comp_encoder,
            num_hiddens_decoder,
            mixing_encoder_kwargs,
            comp_encoder_kwargs,
            decoder_kwargs,
            gmm_kwargs,
            scaler_kwargs,
            obs_name,
            comp_name,
            latent_name,
            module_name,
        )
        if "gaussian" in output_dist:
            raise ValueError("'gaussian' is not supported for CountVAE.")

        self.freq_scale_encoder = ProbabilisticMLP(
            "gaussian",
            num_features,
            1,
            num_hiddens_freq_scale_encoder,
            freq_scale_encoder_kwargs.get("mlp", {}),
            freq_scale_encoder_kwargs.get("probabilistic", {}),
        )
        self.freq_scale_name = freq_scale_name

    def model(
        self,
        obs: torch.Tensor,
        rec_scale: float = 1.0,
        mixing_scale: float = 1.0,
        comp_scale: float = 1.0,
        freq_scale: float = 1.0,
        analytical_kl: bool = False,
    ):
        """Sample from VAE's generative distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            mixing_scale (float): Multiplicative factor for mixing distribution's KL
                regularization term. Default: 1.0
            comp_scale (float): Multiplicative factor for component distribution's KL
                regularization term. Default: 1.0
            freq_scale (float): Multiplicative factor for the frequency scaling term. Default: 1.0
            analytical_kl (bool): Applies closed form KL formulation of GMVAE. Default: False
        """
        pyro.module(self.module_name, self)

        with pyro.plate("N", len(obs)):
            latent = self.gmm(mixing_scale=mixing_scale, comp_scale=comp_scale)

            with scale(scale=freq_scale):
                freq = pyro.sample(self.freq_scale_name, self.freq_scale_prior)

            with scale(scale=rec_scale):
                return pyro.sample(self.obs_name, self.decoder(latent, freq), obs=obs)

    def guide(
        self,
        obs: torch.Tensor,
        rec_scale: float = 1.0,
        mixing_scale: float = 1.0,
        comp_scale: float = 1.0,
        freq_scale: float = 1.0,
        analytical_kl: bool = False,
    ):
        """Sample from VAE's variational distribution.

        Args:
            obs (torch.Tensor): Observed data to be evaluated.
            rec_scale (float): Multiplicative factor for the reconstruction term. Default: 1.0
            mixing_scale (float): Multiplicative factor for mixing distribution's KL
                regularization term. Default: 1.0
            comp_scale (float): Multiplicative factor for component distribution's KL
                regularization term. Default: 1.0
            freq_scale (float): Multiplicative factor for the frequency scaling term. Default: 1.0
            analytical_kl (bool): Applies closed form KL formulation of GMVAE. Default: False
        """
        pyro.module(self.module_name, self)
        obs = self.feature_scaler(obs)

        with pyro.plate("N", len(obs)):
            with scale(scale=mixing_scale):
                mixing_post = self.mixing_encoder(obs)
                comp = pyro.sample(
                    self.gmm.comp_name, mixing_post, **self.gmm.infer_cfg
                )
                if analytical_kl:
                    # Cancels the ratio to use analytical KL
                    mixing_prior = self.gmm.mixing_dist
                    ratio = mixing_prior.log_prob(comp) - mixing_post.log_prob(comp)
                    kl_mixing = kl_divergence(mixing_post, mixing_prior)
                    pyro.factor("ratio_mixing", ratio, has_rsample=False)
                    pyro.factor("kl_mixing", kl_mixing, has_rsample=False)

            with scale(scale=freq_scale):
                freq_post = self.freq_scale_encoder(obs)
                freq = pyro.sample(self.freq_scale_name, freq_post)
                if analytical_kl:
                    # Cancels the ratio to use analytical KL
                    freq_prior = self.freq_scale_prior
                    ratio = freq_prior.log_prob(freq) - freq_post.log_prob(freq)
                    kl_freq = kl_divergence(freq_post, freq_prior)
                    pyro.factor("ratio_freq", ratio, has_rsample=True)
                    pyro.factor("kl_freq", kl_freq, has_rsample=True)

            with scale(scale=comp_scale):
                comp_post = self.comp_encoder(obs, comp)
                latent = pyro.sample(self.gmm.obs_name, comp_post)
                if analytical_kl:
                    # Cancels the ratio to use analytical KL
                    comp_prior = self.gmm.comp_dist(comp)
                    ratio = comp_prior.log_prob(latent) - comp_post.log_prob(latent)
                    kl_latent = kl_divergence(comp_post, comp_prior)
                    pyro.factor("ratio_comp", ratio, has_rsample=True)
                    pyro.factor("kl_latent", kl_latent, has_rsample=True)

        return latent, freq

    def forward(self, obs: torch.Tensor | None = None, decode: bool = False):
        if obs is None:
            latent_dist = self.gmm.distribution
            freq_scale_dist = self.freq_scale_prior
        else:
            obs = self.feature_scaler(obs)
            mixing_dist = self.mixing_encoder(obs)
            latent_dist = self.comp_encoder(obs, mixing_dist.probs)
            freq_scale_dist = self.freq_scale_encoder(obs)

        if not decode:
            return mixing_dist, latent_dist, freq_scale_dist

        if latent_dist.has_rsample:
            latent = latent_dist.rsample()
        else:
            latent = latent_dist.sample()

        if freq_scale_dist.has_rsample:
            freq_scale = freq_scale_dist.rsample()
        else:
            freq_scale = freq_scale_dist.sample()

        return self.decoder(latent, freq_scale)
