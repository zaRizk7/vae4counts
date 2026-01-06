import pprint
import textwrap
from abc import ABC, abstractmethod
from math import ceil, fmod
from typing import Any

import torch


def make_string_of_values(values: list[str], conjuction: str = "and"):
    values = [f"'{value}'" for value in values]
    if len(values) > 1:
        values.insert(-1, conjuction)
    sep = ", " if len(values) > 3 else " "
    string = sep.join(values).replace(f"{conjuction},", conjuction)
    return string


def select_mixture_comp(params: list[torch.Tensor], component: torch.Tensor):
    """Select a Mixture component from one-hot encoded mixture index.

    Args:
        params (list[torch.Tensor]): A list tensor with shape (..., mixture, feature) representing
            the params of mixtures.
        std (torch.Tensor): A tensor with shape (..., mixture, feature) representing
            the mixture of Gaussians standard deviation.
        component (torch.Tensor): A tensor with shape (..., component) representing
            one-hot encoded component index.

    Returns:
        list[torch.Tensor]: A list of tensors with shape (..., feature) representing the selected
            mixture params.
    """
    for i in range(len(params)):
        params[i] = (params[i] * component[..., None]).sum(-2)
    return params


def _check_missing_keys(query: str, available_keys: list[str] | dict[str, Any]):
    if query in available_keys:
        return
    sup_keys = make_string_of_values(list(available_keys))
    raise ValueError(f"'{query}' is not found. Available ones are {sup_keys}.")


class BaseAnnealer(ABC):
    def __init__(self, track_values: bool = False):
        self.track_values = track_values
        self.reset()
        self.repr_mode()

    def __repr__(self):
        extra = self.extra_repr()
        cls_name = self.__class__.__name__

        # If the content is short, keep it on one line
        if len(extra) < 60 and "\n" not in extra:
            return f"{cls_name}({extra})"

        # If it's long, indent the content and put brackets on new lines
        indented = textwrap.indent(extra, "    ")
        return f"{cls_name}(\n{indented}\n)"

    def __call__(self):
        current_value = self.get_current_value()
        if self.track_values:
            self.tracker.append(current_value)
        return current_value

    @property
    def current_step(self):
        return self._current_step

    def step(self):
        self._current_step += 1

    def reset(self):
        self._current_step = 0
        if self.track_values:
            self.tracker = []

    def extra_repr(self) -> str:
        return f"value={self()}, step={self.current_step}"

    def repr_mode(self, mode="state"):
        _check_missing_keys(mode, ("params", "state"))
        self._repr_mode = mode
        return self

    @abstractmethod
    def get_current_value(self) -> float:
        pass


class CyclicalAnnealer(BaseAnnealer):
    def __init__(
        self,
        num_steps: int = 100,
        num_cycles: int = 4,
        bounds: tuple[float, float] = (0, 1),
        annealing_rate: float = 0.5,
        track_values: bool = False,
    ):
        super().__init__(track_values)
        self.num_steps = num_steps
        self.num_cycles = num_cycles
        self.min_val, self.max_val = bounds
        self.annealing_rate = annealing_rate

    @property
    def num_steps_per_cycle(self):
        return self.num_steps / self.num_cycles

    def get_current_value(self):
        tau = fmod(self._current_step, ceil(self.num_steps_per_cycle))
        tau /= self.num_steps_per_cycle
        f_tau = tau / self.annealing_rate

        if tau <= self.annealing_rate:
            return f_tau * (self.max_val - self.min_val) + self.min_val
        return self.max_val

    def extra_repr(self):
        if self._repr_mode == "state":
            return super().extra_repr()

        bounds = (self.min_val, self.max_val)
        return (
            f"num_steps={self.num_steps}, num_cycles={self.num_cycles}, "
            f"bounds={bounds}, annealing_rate={self.annealing_rate}, "
            f"track_values={self.track_values}"
        )


class AnnealerContainer(BaseAnnealer):
    def __init__(self, annealers: dict[str, BaseAnnealer]):
        self.annealers = annealers
        super().__init__()
        self.get_mode()

    def __getitem__(self, keys: str | list[str] | slice) -> dict[str, float] | float:
        if isinstance(keys, slice):
            if not all(getattr(keys, v) is None for v in ("start", "stop", "step")):
                raise ValueError("Only supports [:] to obtain all values.")
            return self[list(self.annealers)]

        if isinstance(keys, str):
            _check_missing_keys(keys, self.annealers)

        if isinstance(keys, str) and self._get_mode == "value":
            return self.annealers[keys]()

        if isinstance(keys, str):
            return self.annealers[keys]

        values = {}
        for k in keys:
            _check_missing_keys(k, self.annealers)
            value = self.annealers[k]
            if self._get_mode == "value":
                value = value()
            values[k] = value
        return values

    def get_current_value(self):
        raise ValueError("Calling directly is not available for AnnealerContainer!")

    def step(self, keys: str | list[str] | None = None):
        super().step()

        if keys is None:
            self.global_step()
            return

        if isinstance(keys, str):
            _check_missing_keys(keys, self.annealers)
            self.annealers[keys].step()
            return

        for k in keys:
            _check_missing_keys(k, self.annealers)
            self.annealers[k].step()

    def global_step(self):
        for v in self.annealers.values():
            v.step()

    def extra_repr(self):
        repr = pprint.pformat(self.annealers, width=1, indent=1).strip("{}")
        repr = textwrap.indent(repr, "    ")
        return repr

    def __repr__(self):
        return f"{self.__class__.__name__}({{\n{self.extra_repr()}\n}})"

    def get_mode(self, mode: str = "value"):
        _check_missing_keys(mode, ("value", "object"))
        self._get_mode = mode
        return self

    def repr_mode(self, mode="state"):
        for v in self.annealers.values():
            v.repr_mode(mode)
        return super().repr_mode(mode)

    def reset(self, keys: str | list[str] | None = None):
        super().reset()
        if keys is None:
            for v in self.annealers.values():
                v.reset()
            return

        if isinstance(keys, str):
            _check_missing_keys(keys, self.annealers)
            self.annealers[keys].reset()

        for k in self.annealers:
            _check_missing_keys(k, self.annealers)
            self.annealers[k].reset()
