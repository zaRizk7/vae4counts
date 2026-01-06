from contextlib import nullcontext

import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import OneHotCategorical, LogNormal, MixtureSameFamily, Normal
from pyro.distributions.constraints import simplex, softplus_positive
from pyro.nn import PyroModule, PyroParam
from pyro.poutine import scale as _scale

from ..activations import nonzero_softplus
from ..utils import select_mixture_comp

__all__ = ["GaussianMixture"]


class GaussianMixture(PyroModule):
    """A PyTorch-based Gaussian Mixture used for Pyro.

    Args:
        num_features (int): Number of features for each mixtures.
        num_components (int): Number of components within the mixture.
        marginal (str): Ways to do marginalization, either "parallel" or "sequential". Default: "parallel"
        init_mixture (str): Initialization method for mixture probs, either "random" or "constant".
            Setting to "constant" initialize the assignment by 1 / num_components. Default: "random"
        init_loc_mul (float): Multiplier to scale the initial mean. Default: 1.0
        obs_name (str): Variable name for the observed variable. Default: "obs"
        comp_name (str): Variable name for the latent variable. Default: "latent"
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
        return {"infer": {"enumerate": self.marginal}}

    @property
    def comp_container(self):
        return LogNormal if self.log else Normal

    @property
    def mixing_dist(self):
        return OneHotCategorical(self.mixing_weights)

    @property
    def comp_dist(self):
        return self.comp_container(self.loc, self.scale).to_event(1)

    @property
    def dist(self):
        return MixtureSameFamily(self.mixing_dist, self.comp_dist)

    def forward(
        self,
        obs: torch.Tensor | None = None,
        kl_scale_mixing: float = 1.0,
        kl_scale_comp: float = 1.0,
    ):
        # Use plate if used as the main model
        with nullcontext() if obs is None else pyro.plate("N", len(obs)):
            with _scale(scale=kl_scale_mixing):
                comp = pyro.sample(self.comp_name, self.mixing_dist, **self.infer_cfg)
            # Manually select the mixture's components to track sampling graph
            loc, scale = select_mixture_comp([self.loc, self.scale], comp)

            with _scale(scale=kl_scale_comp):
                return pyro.sample(
                    self.obs_name, self.comp_container(loc, scale).to_event(1), obs=obs
                )

    def extra_repr(self):
        return (
            f"num_features={self.num_features}, num_components={self.num_components}, "
            f"marginal='{self.marginal}', log={self.log}"
        )
