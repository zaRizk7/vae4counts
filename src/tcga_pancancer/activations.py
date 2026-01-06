import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_string_of_values


def clipped_sigmoid(x: torch.Tensor):
    finfo = torch.finfo(x.dtype)
    return x.sigmoid().clamp(finfo.tiny, 1 - finfo.eps)


def nonzero_softplus(x: torch.Tensor):
    return F.softplus(x) + torch.finfo(x.dtype).eps


class Log1P(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor):
        if self.inplace:
            return input.log1p_()
        return input.log1p()

    def extra_repr(self):
        return f"inplace={self.inplace}"


class CountPerScale(nn.Module):
    def __init__(self, scale=1e6, eps=1e-4, dim=-1):
        super().__init__()
        self.scale = scale
        self.eps = eps
        self.dim = dim

    def forward(self, input: torch.Tensor):
        return F.normalize(input, p=1, dim=self.dim, eps=self.eps) * self.scale

    def extra_repr(self):
        return f"scale={self.scale:,}, eps={self.eps:.5f}, dim={self.dim}"


FEATURE_SCALERS = {
    "identity": nn.Identity,
    "standard": nn.BatchNorm1d,
    "cps": CountPerScale,
    "log1p": Log1P,
}


def make_feature_scaler(scaler: str, **kwargs):
    if scaler != "batch" and "num_features" in kwargs:
        kwargs.pop("num_features")

    if scaler == "batch" and "affine" not in kwargs:
        kwargs["affine"] = False

    try:
        return FEATURE_SCALERS[scaler](**kwargs)
    except KeyError:
        sup_vals = make_string_of_values(list(FEATURE_SCALERS))
        raise ValueError(
            f"Unsupported distribution '{scaler}'! Currently only supports {sup_vals}."
        )
