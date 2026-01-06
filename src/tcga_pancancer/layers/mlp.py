import torch.nn as nn

from ..utils import make_string_of_values

__all__ = ["MLP"]

NORM_LAYER = {"layer": nn.LayerNorm, "batch": nn.BatchNorm1d, "rms": nn.RMSNorm}
ACT_LAYER = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
}


def _make_norm(num_features: int, norm_type: str = "layer", **kwargs) -> nn.Module:
    try:
        return NORM_LAYER[norm_type](num_features, **kwargs)
    except KeyError:
        sup_vals = make_string_of_values(list(NORM_LAYER))
        raise ValueError(
            f"Unsupported norm_type '{norm_type}'! Currently only supports {sup_vals}."
        )


def _make_activation(act_type: str, **kwargs) -> nn.Module:
    try:
        return ACT_LAYER[act_type](**kwargs)
    except KeyError:
        sup_vals = make_string_of_values(list(ACT_LAYER))
        raise ValueError(
            f"Unknown norm_type '{act_type}'! Currently only supports {sup_vals}."
        )


class MLP(nn.Sequential):
    def __init__(
        self,
        num_hiddens: list[int],
        bias: bool = True,
        norm: str = "layer",
        act: str = "relu",
        norm_kwargs: dict = {},
        act_kwargs: dict = {},
    ):
        super().__init__()
        for i in range(1, len(num_hiddens)):
            n_in, n_out = num_hiddens[i - 1], num_hiddens[i]
            self._make_single_block(
                i, n_in, n_out, bias, norm, act, norm_kwargs, act_kwargs
            )

    def _make_single_block(
        self, i, n_in, n_out, bias, norm, act, norm_kwargs, act_kwargs
    ):
        self.add_module(f"lin_{i:0=2d}", nn.Linear(n_in, n_out, bias))
        if act is not None:
            self.add_module(f"act_{i:0=2d}", _make_activation(act, **act_kwargs))
        if norm is not None:
            self.add_module(f"norm_{i:0=2d}", _make_norm(n_out, norm, **norm_kwargs))
