import torch
import torch.nn as nn
from pyro.distributions import (
    Distribution,
    LogNormal,
    Normal,
    OneHotCategorical,
    Poisson,
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
    "make_variational_linear",
    "VariationalMLP",
]


class CategoricalLinear(nn.Linear):
    def forward(self, input):
        logits = super().forward(input)
        return OneHotCategorical(logits=logits)


class GaussianLinear(nn.Linear):
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
    def __init__(
        self,
        in_features,
        out_features,
        num_components,
        condition="select",
        bias=True,
        constant_scale=False,
        log=False,
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
            self.latent_proj = nn.Linear(num_components, out_features)
        else:
            self.register_module("latent_proj", None)

    @property
    def container(self):
        return LogNormal if self.log else Normal

    @property
    def reshape_size(self):
        if self.condition == "additive":
            raise ValueError("Only available when condition='select'.")
        return (-1, self.num_components, self.out_features // self.num_components)

    def forward(
        self, input: torch.Tensor, component: torch.Tensor | None = None
    ) -> Normal:
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
        out_features = self.out_features // self.num_components
        if not self.constant_scale:
            out_features //= 2
        return (
            f"in_features={self.in_features}, out_features={out_features}, "
            f"num_components={self.num_components}, condition='{self.condition}', "
            f"bias={self.bias is not None}, constant_scale={self.constant_scale}, "
            f"log={self.log}"
        )


class PoissonLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> Poisson:
        log_rate = super().forward(input)
        rate = nonzero_softplus(log_rate)
        return Poisson(rate).to_event(1)


class ZIPoissonLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features * 2, bias, device, dtype)

    def forward(self, input: torch.Tensor) -> ZeroInflatedPoisson:
        output = super().forward(input)
        log_rate, logit = output.chunk(2, -1)
        rate = nonzero_softplus(log_rate)
        return ZeroInflatedPoisson(rate, gate_logits=logit).to_event(1)

    def extra_repr(self):
        src = f"out_features={self.out_features}"
        tgt = f"out_features={self.out_features // 2}"
        return super().extra_repr().replace(src, tgt)


VARIATIONAL_LINEAR_LAYERS = {
    "categorical": CategoricalLinear,
    "gaussian": GaussianLinear,
    "gaussian_mixture": GaussianMixtureLinear,
    "poisson": PoissonLinear,
    "zi_poisson": ZIPoissonLinear,
}


def make_variational_linear(
    distribution: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    device=None,
    dtype=None,
    **kwargs,
) -> nn.Linear:
    if distribution != "gaussian" and "constant_scale" in kwargs:
        kwargs.pop("constant_scale")

    try:
        return VARIATIONAL_LINEAR_LAYERS[distribution](
            in_features, out_features, bias=bias, device=device, dtype=dtype, **kwargs
        )
    except KeyError:
        sup_vals = make_string_of_values(list(VARIATIONAL_LINEAR_LAYERS))
        raise KeyError(
            f"Unsupported distribution '{distribution}'! Currently only supports {sup_vals}."
        )


class VariationalMLP(nn.Module):
    def __init__(
        self,
        distribution: str,
        in_features: int,
        out_features: int,
        num_hiddens: list[int] | None = None,
        mlp_kwargs: dict = {},
        variational_kwargs: dict = {},
    ):
        super().__init__()
        if num_hiddens is None:
            last_hidden = in_features
            self.mlp = nn.Identity()
        else:
            last_hidden = num_hiddens[-1]
            self.mlp = MLP([in_features] + num_hiddens, **mlp_kwargs)

        self.variational = make_variational_linear(
            distribution, last_hidden, out_features, **variational_kwargs
        )

    def forward(self, input, *args, **kwargs) -> Distribution:
        hidden = self.mlp(input)
        return self.variational(hidden, *args, **kwargs)
