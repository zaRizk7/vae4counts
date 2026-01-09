import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_string_of_values


def clipped_sigmoid(x: torch.Tensor):
    """Sigmoid but with smallest minmax clamp possible for `x`'s dtype.

    Args:
        x (torch.Tensor): Value to apply the clipped sigmoid to.
    """
    finfo = torch.finfo(x.dtype)
    return x.sigmoid().clamp(finfo.tiny, 1 - finfo.eps)


def nonzero_softplus(x: torch.Tensor):
    """Softplus but with smallest clamp possible for `x`'s dtype to
    prevent zero.

    Args:
        x (torch.Tensor): Value to apply the clipped sigmoid to.
    """
    return F.softplus(x) + torch.finfo(x.dtype).eps


class Log1P(nn.Module):
    """A wrapper to apply `log1p(x)` as a layer.

    Args:
        inplace (bool): Applies in-place ops or not. Default: False
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor):
        if self.inplace:
            return input.log1p_()
        return input.log1p()

    def extra_repr(self):
        return f"inplace={self.inplace}"


class CountPerScale(nn.Module):
    """A wrapper to apply count per scale to positive values as a layer. It will
    apply normalization on the feature dimension. By default, this module will
    apply count per-million (cpm), commonly seen for RNA-seq data.

    Args:
        scale (float): The multiplicative scale used after normalization. Default: 1e6
        eps (float): Small value to prevent zero division. Default: 1e-4
        dim (int): Feature dimension to normalize. Default: -1
    """

    def __init__(self, scale: float = 1e6, eps: float = 1e-4, dim: int = -1):
        super().__init__()
        self.scale = scale
        self.eps = eps
        self.dim = dim

    def forward(self, input: torch.Tensor):
        return F.normalize(input, p=1, dim=self.dim, eps=self.eps) * self.scale

    def extra_repr(self):
        return f"scale={self.scale:,}, eps={self.eps:.5f}, dim={self.dim}"


class Log1PCPS(nn.Sequential):
    """A wrapper that applied `log1p(cps(x))` as a layer.

    Args:
        scale (float): The multiplicative scale used after normalization. Default: 1e6
        eps (float): Small value to prevent zero division. Default: 1e-4
        dim (int): Feature dimension to normalize. Default: -1
    """

    def __init__(self, scale=1e6, eps=1e-4, dim=-1):
        super().__init__()
        self.add_module("log1p", Log1P())
        self.add_module("cps", CountPerScale(scale, eps, dim))


FEATURE_SCALERS = {
    "identity": nn.Identity,
    "standard": nn.BatchNorm1d,
    "cps": CountPerScale,
    "log1p": Log1P,
    "log1pcps": Log1PCPS,
}


def make_feature_scaler(
    scaler: str, **kwargs
) -> nn.Identity | nn.BatchNorm1d | CountPerScale | Log1P | Log1PCPS:
    """Factory function to initialize feature scaler. Available ones are:
        - "identity": Identity function (no scaling).
        - "standard": Standardization, uses batch norm without any affine by default.
        - "cps": Count per scale, normalizes by feature dimension.
        - "log1p": Applies stable version of `log(1+x)`.
        - "log1pcps": Applies `log(1+cps(x))`.

    For 'standard', affine can be enabled by specifying the params for `nn.BatchNorm1d`
    in `kwargs`.

    Args:
        scaler (str): Feature scaler to be initialized.
        **kwargs: Additional arguments to initialize the layer.

    Returns:
        nn.Identity | nn.BatchNorm1d | CountPerScale | Log1P | Log1PCPS:
            Initialized feature scaling layer.
    """
    if scaler != "standard" and "num_features" in kwargs:
        kwargs.pop("num_features")

    if scaler == "standard" and "affine" not in kwargs:
        kwargs["affine"] = False

    try:
        return FEATURE_SCALERS[scaler](**kwargs)
    except KeyError:
        sup_vals = make_string_of_values(list(FEATURE_SCALERS))
        raise ValueError(
            f"Unsupported distribution '{scaler}'! Currently only supports {sup_vals}."
        )
