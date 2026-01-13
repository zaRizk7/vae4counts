from contextlib import nullcontext

import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import (
    Categorical,
    LogNormal,
    MixtureSameFamily,
    Normal,
    OneHotCategorical,
)
from pyro.distributions.constraints import simplex, softplus_positive
from pyro.nn import PyroModule, PyroParam
from pyro.poutine import scale as _scale

from ..utils import select_mixture_comp

__all__ = ["GaussianMixture"]


class GaussianMixture(PyroModule):
    """A PyTorch-based Gaussian Mixture used for Pyro. The module can be used as a standalone
    model or a submodule for the `GMVAE`. Note that this is an MLE Gaussian Mixture, not the
    variational version. Hence, this implementation does not implement a variational guide
    function.

    Args:
        num_features (int): Number of features for each mixtures.
        num_components (int): Number of components within the mixture.
        marginal (str): Ways to do marginalization, either "parallel" or "sequential". Default: "parallel"
        init_mixture (str): Initialization method for mixture probs, either "random" or "constant".
            Setting to "constant" initialize the assignment by 1 / num_components. Default: "random"
        init_loc_mul (float): Multiplier to scale the initial mean. Default: 1.0
        freeze_mixing_weights (bool): Prevent parametrization of the mixture's mixing weights. Default: True
        log (bool): Whether to use log-Gaussian instead of Gaussian. Default: False
        obs_name (str): Variable name for the observed variable. Default: "obs"
        comp_name (str): Variable name for the component latent variable. Default: "comp"
        module_name (str): A name to register the module to Pyro. Normally used if this module
            only used for another module extensions. Default: "gmm"
    """

    def __init__(
        self,
        num_features: int,
        num_components: int,
        marginal: str = "parallel",
        init_mixture: str = "constant",
        init_loc_mul: float = 1.0,
        log: bool = False,
        freeze_mixing_weights: bool = True,
        obs_name: str = "obs",
        comp_name: str = "comp",
        module_name: str = "gmm",
    ):
        super().__init__()
        self.num_features = num_features
        self.num_components = num_components
        self.marginal = marginal
        self.init_mixture = init_mixture
        self.init_loc_mul = init_loc_mul
        self.log = log
        self.freeze_mixing_weights = freeze_mixing_weights
        self.obs_name = obs_name
        self.comp_name = comp_name
        self.module_name = module_name
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Resets the parameters available"""
        # Define param shape
        mixing_logits = torch.empty(self.num_components)
        loc = torch.empty(self.num_components, self.num_features)
        scale_sp_inv = torch.empty(self.num_components, self.num_features)
        # Use Glorot's initialization on single vector dimension.
        if self.init_mixture == "random":
            nn.init.normal_(mixing_logits, std=0.1)
            mixing_logits = mixing_logits.softmax(0)
        else:
            nn.init.ones_(mixing_logits).div_(self.num_components)
        # multiply to have more spread out mean
        nn.init.normal_(loc, 0, self.init_loc_mul / 3**0.5)
        # Init to be roughly std=1.0 when softplus is applied
        nn.init.normal_(scale_sp_inv, 0.5413, 0.1)
        # Register parameters
        # self.mixing_weights = PyroParam(mixing_logits, simplex)
        if self.freeze_mixing_weights:
            self.mixing_weights = nn.Buffer(mixing_logits)
        else:
            self.mixing_weights = PyroParam(mixing_logits, simplex)
        self.loc = PyroParam(loc)
        self.scale = PyroParam(scale_sp_inv, softplus_positive)

    @property
    def infer_cfg(self):
        """Mapping for the component inference"""
        return {"infer": {"enumerate": self.marginal}}

    @property
    def comp_container(self):
        """Container for the component distribution."""
        return LogNormal if self.log else Normal

    @property
    def mixing_dist(self):
        """Distribution for the mixing weights."""
        return OneHotCategorical(self.mixing_weights)

    def comp_dist(self, comp: torch.Tensor | None = None) -> Normal | LogNormal:
        """Fetch component distribution. If `comp` is specified, it will
        select the specific component's distribution.

        Args:
            comp (torch.Tensor | None): A tensor with shape (..., component) representing
                one-hot encoded component index. Default: None

        Returns:
            (Normal | LogNormal): Distribution of the selected component.
        """
        # Manually select the mixture's components to track sampling graph
        loc, scale = self.loc, self.scale
        if comp is not None:
            loc, scale = select_mixture_comp([loc, scale], comp)
        return self.comp_container(loc, scale).to_event(1)

    @property
    def dist(self) -> MixtureSameFamily:
        """Complete distribution for the Gaussian mixture."""
        return MixtureSameFamily(Categorical(self.mixing_weights), self.comp_dist())

    def forward(
        self,
        obs: torch.Tensor | None = None,
        comp: torch.Tensor | None = None,
        mixing_scale: float = 1.0,
        comp_scale: float = 1.0,
    ):
        """Sampling the mixture distribution model.

        Args:
            obs (torch.Tensor | None): The observed value. If not specified,
                it is assumed that the model is used as another model's module. Default: None
            comp (torch.Tensor | None): If it is not None, comp will be assumed to be observed.
                Modifying the behavior to mimic Naive Bayes. Default: None
            mixing_scale (float): The scaling for the mixing log-likelihood. In SVI, it will be used
                for the KL divergence between the mixture and another distribution. Default: 1.0
            comp_scale (float): The scaling for the component log-likelihood. In SVI, it will be used
                for the KL divergence between the component and another distribution. Default: 1.0
        """
        if obs is not None or comp is not None:
            pyro.module(self.module_name, self)

        # Use plate if used as the main model
        with nullcontext() if obs is None else pyro.plate("N", len(obs)):
            with _scale(scale=mixing_scale):
                comp = pyro.sample(
                    self.comp_name, self.mixing_dist, obs=comp, **self.infer_cfg
                )

            with _scale(scale=comp_scale):
                return pyro.sample(self.obs_name, self.comp_dist(comp), obs=obs)

    def extra_repr(self):
        return (
            f"num_features={self.num_features}, num_components={self.num_components}, "
            f"marginal='{self.marginal}', log={self.log}"
        )

    @torch.no_grad()
    @staticmethod
    def fit_from_data(
        obs: torch.Tensor,
        comp: torch.Tensor | None = None,
        freeze_params: bool = True,
        **kwargs,
    ) -> GaussianMixture:
        # Let it be parametrized
        kwargs["freeze_mixing_weights"] = False

        # Basically naive bayes like way
        mixing_weights = torch.ones(1)
        if comp is not None:
            mixing_weights = F.normalize(comp.bincount(), p=1, dim=-1)
        else:
            comp = torch.zeros(len(obs))

        if kwargs.get("log", False):
            obs = obs.log().clamp_min(torch.finfo(torch.float).eps)

        if obs.dim() == 1:
            obs = obs.unsqueeze(-1)

        # Fit MLE
        loc = torch.empty(len(mixing_weights), *obs.shape[1:])
        scale = torch.empty_like(loc)
        for i, c in enumerate(comp.unique()):
            loc[i] = obs[c == comp].mean(0)
            scale[i] = obs[c == comp].std(0)

        # Scale is being softplus inverted to support pyro's
        # parametrization
        # scale + log1p(-exp(-scale))
        params = {
            "mixing_weights": mixing_weights,
            "loc": loc,
            "scale": scale + scale.neg().exp().neg().log1p(),
        }
        params = {f"{k}_unconstrained": v for k, v in params.items()}

        # Initialize
        gmm = GaussianMixture(obs.shape[-1], len(mixing_weights), **kwargs)
        gmm.load_state_dict(params)

        # Use case: Empirical Bayes prior
        if freeze_params:
            for param in gmm.parameters():
                param.requires_grad_(False)

        return gmm
