import torch.nn as nn

from ..utils import make_string_of_values

__all__ = ["MLP"]

NORM_LAYER = {
    "identity": nn.Identity,
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
    "rms": nn.RMSNorm,
}
ACT_LAYER = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
}


def _make_norm(
    num_features: int, norm_type: str = "layer", **kwargs
) -> nn.Identity | nn.LayerNorm | nn.BatchNorm1d | nn.RMSNorm:
    """Factory function to initialize normalization layer. Available ones are:
    - "identity": Identity function (no normalization).
    - "layer": Feature-wise standardization.
    - "batch": Batch-wise standardization.
    - "rms": Applies feature-wise root mean square normalization.

    Args:
        num_features (int): Number of features.
        norm_type (str): Normalization layer type. Default: "layer"
        **kwargs: Additional params for the chosen normalization layer.

    Returns:
        nn.Identity | nn.LayerNorm | nn.BatchNorm1d | nn.RMSNorm:
            Initialized normalization layer.

    Raises:
        ValueError: Unsupported `norm_type`.
    """
    try:
        return NORM_LAYER[norm_type](num_features, **kwargs)
    except KeyError:
        sup_vals = make_string_of_values(list(NORM_LAYER))
        raise ValueError(
            f"Unsupported norm_type '{norm_type}'! Currently only supports {sup_vals}."
        )


def _make_activation(
    act_type: str, **kwargs
) -> nn.Identity | nn.ReLU | nn.GELU | nn.SiLU | nn.Softplus:
    """Factory function to initialize piecewise activation layer. Available ones are:
    - "identity": Identity function (no activation).
    - "relu": Applies Rectified Linear Unit (ReLU) `max(0, x)`.
    - "gelu": Applies Gaussian Error Linear Unit `x * sigmoid(1.702*x)`.
    - "silu" or "swish": Applies Sigmoid Linear Unit or Swish activation `x * sigmoid(x)`.
    - "softplus": Applies `log(1+e^x)` to each element.

    Returns:
        nn.Identity | nn.ReLU | nn.GELU | nn.SiLU | nn.Softplus:
            Initialized piecewise activation layer.

    Raises:
        ValueError: Unsupported `act_type`.
    """
    try:
        return ACT_LAYER[act_type](**kwargs)
    except KeyError:
        sup_vals = make_string_of_values(list(ACT_LAYER))
        raise ValueError(
            f"Unknown norm_type '{act_type}'! Currently only supports {sup_vals}."
        )


class MLP(nn.Sequential):
    """Multi-layer Perceptron (MLP) layer with repeated blocks containing
    `Linear -> Activation -> Normalization`.

    Args:
        num_hiddens (list[int]): List of number of hidden layers' sizes.
            The first and last layer will be the input and output size.
        bias (bool): Adds additive bias to all hidden layers. Default: True
        norm (str): Normalization layer to use in a single block.
            Available ones are "identity", "layer", "batch", and "rms".
            Default: "layer".
        act (str): Piecewise activation layer to use in a single block.
            Available ones are "identity", "relu", "gelu", "silu", "swish", and
            "softplus". Default: "relu"
        norm_kwargs (dict): Additional params for the normalization layer. Default: {}
        act_kwargs (dict): Additional params for the activation layer. Default: {}
    """

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
        self,
        i: int,
        n_in: int,
        n_out: int,
        bias: bool,
        norm: str,
        act: str,
        norm_kwargs: dict,
        act_kwargs: dict,
    ):
        """Produce a single block of MLP.

        Args:
            i (int): Index for the i-th block.
            n_in (int): Number of inputs.
            n_out (out): Number of outputs.
            bias (bool): Include additive bias.
            norm (str): Normalization layer type.
            act (str): Piecewise activation layer type.
            norm_kwargs (str): Additional params for normalization layer.
            act_kwargs (str): Additional params for activation layer.
        """
        self.add_module(f"lin_{i:0=2d}", nn.Linear(n_in, n_out, bias))
        if act is not None:
            self.add_module(f"act_{i:0=2d}", _make_activation(act, **act_kwargs))
        if norm is not None:
            self.add_module(f"norm_{i:0=2d}", _make_norm(n_out, norm, **norm_kwargs))
