import torch
import torch.nn as nn
from pyro.distributions import (
    Distribution,
    LogNormal,
    NegativeBinomial,
    Normal,
    OneHotCategorical,
    Poisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)

from ..activations import nonzero_softplus
from ..utils import make_string_of_values, select_mixture_comp
from .mlp import MLP

__all__ = [
    "CategoricalLinear",
    "GaussianLinear",
    "PoissonLinear",
    "ZIPoissonLinear",
    "make_probabilistic_linear",
    "ProbabilisticMLP",
]


def _rate_fn(log_rate: torch.Tensor, log_scale: torch.Tensor | None = None):
    """Applies softplus if `log_scale` is not defined, else do scVI style scaling."""
    if log_scale is None:
        return nonzero_softplus(log_rate)
    eps = torch.finfo(log_scale.dtype).eps
    return (log_rate.log_softmax(-1) + log_scale + eps).exp()


class CategoricalLinear(nn.Linear):
    """An extension of `nn.Linear` that outputs a categorical distribution
    given an input.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
    """

    def forward(self, input: torch.Tensor) -> OneHotCategorical:
        logits = super().forward(input)
        return OneHotCategorical(logits=logits)


class GaussianLinear(nn.Linear):
    """An extension of `nn.Linear` that outputs a Gaussian distribution
    given an input.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
        constant_scale (bool): Assumes that the standard deviation is equal to one.
            Default: False
        log (bool): If True, uses log-Gaussian instead of Gaussian. Default: False
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        constant_scale=False,
        log=False,
        device=None,
        dtype=None,
    ):
        num_outputs = 1 if constant_scale else 2
        super().__init__(in_features, out_features * num_outputs, bias, device, dtype)
        self.constant_scale = constant_scale
        self.log = log

    @property
    def container(self):
        return LogNormal if self.log else Normal

    def forward(self, input: torch.Tensor) -> Normal:
        loc, scale = super().forward(input), 1.0
        if not self.constant_scale:
            loc, log_scale = loc.chunk(2, -1)
            scale = nonzero_softplus(log_scale)

        return self.container(loc, scale).to_event(1)

    def extra_repr(self):
        out_features = self.out_features
        if not self.constant_scale:
            out_features //= 2
        return (
            f"in_features={self.in_features}, out_features={out_features}, "
            f"bias={self.bias is not None}, constant_scale={self.constant_scale}, "
            f"log={self.log}"
        )


class GaussianMixtureLinear(nn.Linear):
    """An extension of `nn.Linear` that outputs a Gaussian mixture distribution
    given an input and a one-hot encoded component index to condition on a component.
    There are two ways to condition:
        - "select": Outputs over num_components of distribution and selects with
            the one-hot encoded index by dot product over the component dim.
        - "additive": Adds an additional linear layer with the same out_features and
            adds the embedding to the output before any link function is applied.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        num_components (int): Number of components in the mixture.
        condition (str): Conditioning approach to select a component. Either
            "select" or "additive". Default: "select"
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
        constant_scale (bool): Assumes that the standard deviation is equal to one.
            Default: False
        log (bool): If True, uses log-Gaussian instead of Gaussian. Default: False
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_components: int,
        condition: str = "select",
        bias: bool = True,
        constant_scale: bool = False,
        log: bool = False,
        device=None,
        dtype=None,
    ):
        num_outputs = 1 if constant_scale else 2
        if condition not in {"select", "additive"}:
            raise KeyError("Invalid condition, only supports 'select' and 'additive'.")
        out_features = out_features * num_outputs
        if condition == "select":
            out_features *= num_components
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_components = num_components
        self.condition = condition
        self.constant_scale = constant_scale
        self.log = log

        if condition == "additive":
            # works the same as nn.Embedding, just with one-hot encoding
            self.component_proj = nn.Linear(num_components, out_features)
        else:
            self.register_module("component_proj", None)

    @property
    def container(self):
        """Container for the component distribution"""
        return LogNormal if self.log else Normal

    @property
    def reshape_size(self):
        """Expected reshape size for the original linear output to obtain output of
        (-1, components, features). Only used when condition is set to "select".
        """
        if self.condition == "additive":
            raise ValueError("Only available when condition='select'.")
        return (-1, self.num_components, self.out_features // self.num_components)

    def forward(
        self, input: torch.Tensor, component: torch.Tensor | None = None
    ) -> Normal | LogNormal:
        loc, scale = super().forward(input), 1.0

        if self.condition == "additive" and component is not None:
            loc = loc + self.component_proj(component)
        elif component is not None:
            loc = loc.reshape(*self.reshape_size)
            (loc,) = select_mixture_comp([loc], component)

        if not self.constant_scale:
            loc, log_scale = loc.chunk(2, -1)
            scale = nonzero_softplus(log_scale)

        return self.container(loc, scale).to_event(1)

    def extra_repr(self):
        out_features = self.out_features
        if self.condition == "select":
            out_features //= self.num_components

        if not self.constant_scale:
            out_features //= 2

        return (
            f"in_features={self.in_features}, out_features={out_features}, "
            f"num_components={self.num_components}, condition='{self.condition}', "
            f"bias={self.bias is not None}, constant_scale={self.constant_scale}, "
            f"log={self.log}"
        )


class NegativeBinomialLinear(nn.Linear):
    """An extension of `nn.Linear` that outputs a Negative Binomial distribution
    given an input.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
        shared_gate (bool): If True, uses a parametrized dispersion gate instead
            from output. Default: False
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        shared_gate: bool = False,
        device=None,
        dtype=None,
    ):
        num_outputs = 1 if shared_gate else 2
        out_features *= num_outputs
        super().__init__(in_features, out_features, bias, device, dtype)
        if shared_gate:
            self.logits = nn.Parameter(torch.empty(out_features // num_outputs))
        else:
            self.register_parameter("logits", None)
        self.reset_parameters(True)

    @torch.no_grad()
    def reset_parameters(self, reset_after_inherit: bool = False):
        # Workaround when inheriting nn.Linear
        if not reset_after_inherit:
            return
        super().reset_parameters()
        if self.logits is not None:
            nn.init.normal_(self.logits, std=0.05)

    def forward(
        self, input: torch.Tensor, log_scale: torch.Tensor | None = None
    ) -> NegativeBinomial:
        log_rate = super().forward(input)
        logits = self.logits
        if logits is None:
            log_rate, logits = log_rate.chunk(2, -1)
        rate = _rate_fn(log_rate, log_scale)
        return NegativeBinomial(rate, logits=logits).to_event(1)

    def extra_repr(self):
        shared_gate = self.logits is not None
        out_features = self.out_features
        if not shared_gate:
            out_features //= 2
        return (
            f"in_features={self.in_features}, out_features={out_features}, "
            f"bias={self.bias is not None}, shared_gate={shared_gate}"
        )


class PoissonLinear(nn.Linear):
    """An extension of `nn.Linear` that outputs a Poisson distribution
    given an input.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
    """

    def forward(
        self, input: torch.Tensor, log_scale: torch.Tensor | None = None
    ) -> Poisson:
        log_rate = super().forward(input)
        rate = _rate_fn(log_rate, log_scale)
        return Poisson(rate).to_event(1)


class ZIPoissonLinear(nn.Linear):
    """An extension of `nn.Linear` that outputs a Zero-Inflated Poisson distribution
    given an input.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
        shared_gate (bool): If True, uses a parametrized zero inflation gate instead
            from output. Default: False
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        shared_gate=False,
        device=None,
        dtype=None,
    ):
        num_outputs = 1 if shared_gate else 2
        out_features *= num_outputs
        super().__init__(in_features, out_features, bias, device, dtype)
        if shared_gate:
            self.gate_logits = nn.Parameter(torch.empty(out_features // num_outputs))
        else:
            self.register_parameter("gate_logits", None)
        self.reset_parameters(True)

    @torch.no_grad()
    def reset_parameters(self, reset_after_inherit: bool = False):
        # Workaround when inheriting nn.Linear
        if not reset_after_inherit:
            return
        super().reset_parameters()
        if self.gate_logits is not None:
            nn.init.normal_(self.gate_logits, std=0.05)

    def forward(
        self, input: torch.Tensor, log_scale: torch.Tensor | None = None
    ) -> ZeroInflatedPoisson:
        log_rate = super().forward(input)
        gate_logits = self.gate_logits
        if gate_logits is None:
            log_rate, gate_logits = log_rate.chunk(2, -1)
        rate = _rate_fn(log_rate, log_scale)
        return ZeroInflatedPoisson(rate, gate_logits=gate_logits).to_event(1)

    def extra_repr(self):
        shared_gate = self.gate_logits is not None
        out_features = self.out_features
        if not shared_gate:
            out_features //= 2
        return (
            f"in_features={self.in_features}, out_features={out_features}, "
            f"bias={self.bias is not None}, shared_gate={shared_gate}"
        )


class ZINegativeBinomialLinear(nn.Linear):
    """An extension of `nn.Linear` that outputs a Negative Binomial distribution
    given an input.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
        shared_nb_gate (bool): If True, uses a parametrized dispersion gate instead
            from output. Default: False
        shared_nb_gate (bool): If True, uses a parametrized zero-inflation gate instead
            from output. Default: False
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        shared_nb_gate: bool = False,
        shared_zi_gate: bool = False,
        device=None,
        dtype=None,
    ):
        num_outputs = 1
        num_outputs += 1 if not shared_nb_gate else 0
        num_outputs += 1 if not shared_zi_gate else 0
        out_features *= num_outputs
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_outputs = num_outputs
        if shared_nb_gate:
            self.nb_gate_logits = nn.Parameter(torch.empty(out_features // num_outputs))
        else:
            self.register_parameter("nb_gate_logits", None)
        if shared_zi_gate:
            self.zi_gate_logits = nn.Parameter(torch.empty(out_features // num_outputs))
        else:
            self.register_parameter("zi_gate_logits", None)
        self.reset_parameters(True)

    @torch.no_grad()
    def reset_parameters(self, reset_after_inherit: bool = False):
        # Workaround when inheriting nn.Linear
        if not reset_after_inherit:
            return
        super().reset_parameters()
        if self.nb_gate_logits is not None:
            nn.init.normal_(self.nb_gate_logits, std=0.05)
        if self.zi_gate_logits is not None:
            nn.init.normal_(self.zi_gate_logits, std=0.05)

    def forward(
        self, input: torch.Tensor, log_scale: torch.Tensor | None = None
    ) -> ZeroInflatedNegativeBinomial:
        log_rate = super().forward(input)
        nb_logits, zi_logits = self.nb_gate_logits, self.zi_gate_logits

        if nb_logits is None and zi_logits is None:
            log_rate, nb_logits, zi_logits = log_rate.chunk(self.num_outputs, -1)
        elif nb_logits is None and zi_logits is not None:
            log_rate, nb_logits = log_rate.chunk(self.num_outputs, -1)
        elif nb_logits is not None and zi_logits is None:
            log_rate, zi_logits = log_rate.chunk(self.num_outputs, -1)
        rate = _rate_fn(log_rate, log_scale)

        return ZeroInflatedNegativeBinomial(
            rate, logits=nb_logits, gate_logits=zi_logits
        ).to_event(1)

    def extra_repr(self):
        shared_nb_gate = self.nb_gate_logits is not None
        shared_zi_gate = self.zi_gate_logits is not None
        out_features = self.out_features // self.num_outputs
        return (
            f"in_features={self.in_features}, out_features={out_features}, "
            f"bias={self.bias is not None}, shared_nb_gate={shared_nb_gate} "
            f"shared_zi_gate={shared_zi_gate}"
        )


PROBABILISTIC_LINEAR_LAYERS = {
    "categorical": CategoricalLinear,
    "gaussian": GaussianLinear,
    "gaussian_mixture": GaussianMixtureLinear,
    "negative_binomial": NegativeBinomialLinear,
    "poisson": PoissonLinear,
    "zi_poisson": ZIPoissonLinear,
    "zi_nb": ZINegativeBinomialLinear,
}


def make_probabilistic_linear(
    distribution: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    device=None,
    dtype=None,
    **kwargs,
) -> (
    CategoricalLinear
    | GaussianLinear
    | GaussianMixtureLinear
    | NegativeBinomialLinear
    | PoissonLinear
    | ZIPoissonLinear
):
    """Factory function to initialize feature scaler. Available ones are:
        - "categorical": Models categorical distribution.
        - "gaussian": Models Gaussian distribution.
        - "gaussian_mixture": Models mixture of Gaussian distribution.
        - "negative_binomial": Models negative binomial distribution.
        - "poisson": Models Poisson distribution.
        - "zi_poisson": Models zero-inflated Poisson distribution.

    Args:
        distribution (str): Chosen modeling distribution.
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias.
            Default: True
        **kwargs: Additional params for the specific probabilistic layer.
    """
    if distribution != "gaussian" and "constant_scale" in kwargs:
        kwargs.pop("constant_scale")

    try:
        return PROBABILISTIC_LINEAR_LAYERS[distribution](
            in_features, out_features, bias=bias, device=device, dtype=dtype, **kwargs
        )
    except KeyError:
        sup_vals = make_string_of_values(list(PROBABILISTIC_LINEAR_LAYERS))
        raise KeyError(
            f"Unsupported distribution '{distribution}'! Currently only supports {sup_vals}."
        )


class ProbabilisticMLP(nn.Module):
    """An extension of MLP that outputs a probability distribution. Available ones are:
        - "categorical": Models categorical distribution.
        - "gaussian": Models Gaussian distribution.
        - "gaussian_mixture": Models mixture of Gaussian distribution.
        - "poisson": Models Poisson distribution.
        - "zi_poisson": Models Zero-Inflated Poisson distribution.

    Args:
        distribution (str): Chosen modeling distribution.
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        num_hiddens (list[int] | None): List of hidden features to form a MLP between
            in_features and out_features. If None, then reduces to a linear probabilistic
            layer. Default: None
        mlp_kwargs (dict): Additional params to initialize MLP. Default: {}
        proba_kwargs (dict): Additional params to initialize probabilistic layer. Default: {}
    """

    def __init__(
        self,
        distribution: str,
        in_features: int,
        out_features: int,
        num_hiddens: list[int] | None = None,
        mlp_kwargs: dict = {},
        proba_kwargs: dict = {},
    ):
        super().__init__()
        if num_hiddens is None:
            last_hidden = in_features
            self.mlp = nn.Identity()
        else:
            last_hidden = num_hiddens[-1]
            self.mlp = MLP([in_features] + num_hiddens, **mlp_kwargs)

        self.probs_proj = make_probabilistic_linear(
            distribution, last_hidden, out_features, **proba_kwargs
        )

    def forward(self, input, *args, **kwargs) -> Distribution:
        hidden = self.mlp(input)
        return self.probs_proj(hidden, *args, **kwargs)
