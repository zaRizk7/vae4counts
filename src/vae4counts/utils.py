import pprint
import textwrap
from abc import ABC, abstractmethod
from math import ceil, fmod
from typing import Any

import torch


def make_string_of_values(values: list[str], conjuction: str = "and"):
    """Generates string of human readable string from list of values

    Args:
        values (list[str]): List of string values to be combined.
        conjunction (str): Conjunction for the last element. Default: "and"

    Returns:
        str: The merged values with conjunction on the last element.
    """
    values = [f"'{value}'" for value in values]
    if len(values) > 1:
        values.insert(-1, conjuction)
    sep = ", " if len(values) > 3 else " "
    string = sep.join(values).replace(f"{conjuction},", conjuction)
    return string


def select_mixture_comp(
    params: list[torch.Tensor], component: torch.Tensor
) -> list[torch.Tensor]:
    """Select a Mixture component from one-hot encoded mixture index.

    Args:
        params (list[torch.Tensor]): A list tensor with shape (..., component, feature) representing
            the params of mixture component.
        component (torch.Tensor): A tensor with shape (..., component) representing
            one-hot encoded component index.

    Returns:
        list[torch.Tensor]: A list of tensors with shape (..., feature) representing the selected
            component params.
    """
    for i in range(len(params)):
        params[i] = (params[i] * component[..., None]).sum(-2)
    return params


def _check_missing_keys(query: str, available_keys: list[str] | dict[str, Any]):
    """Checks if the query is not on the available keys.

    Args:
        query (str):
    """
    if query in available_keys:
        return
    sup_keys = make_string_of_values(list(available_keys))
    raise ValueError(f"'{query}' is not found. Available ones are {sup_keys}.")


class BaseAnnealer(ABC):
    """Base class for annealing values.

    Args:
        track_values (bool): Track values produced during annealing. Default: False
    """

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
        """Get the annealed value at currrent time step. If `track_values=True`,
        will also add the current value to tracker.

        Returns:
            float: Annealed value.
        """
        current_value = self.get_current_value()
        if self.track_values:
            self.tracker.append(current_value)
        return current_value

    @property
    def current_step(self):
        """Current step of the annealer."""
        return self._current_step

    def step(self, increment: int = 1):
        """Increment the current step.

        Args:
            increment (int): Value for increment. Default: 1
        """
        self._current_step += increment

    def reset(self):
        """Resets the current step to zero. If `track_values=True`,
        resets the value tracker as well.
        """
        self._current_step = 0
        if self.track_values:
            self.tracker = []

    def extra_repr(self) -> str:
        """Gets the specified object information."""
        return f"value={self()}, step={self.current_step}"

    def repr_mode(self, mode="state"):
        """Sets the information given on the repr."""
        _check_missing_keys(mode, ("params", "state"))
        self._repr_mode = mode
        return self

    @abstractmethod
    def get_current_value(self) -> float:
        """Gets the current value."""
        pass


class CyclicalAnnealer(BaseAnnealer):
    """
    Cyclical annealer to gradually increase the value over
    steps from the lower to upper bound until it reaches the rate fraction.
    Then, resets the value to the lower bound and is repeater per cycle.

    Args:
        num_steps (int): Total number of steps. Default: 100
        num_cycles (int): Number of cycles to anneal values. Default: 4
        bounds (tuple[float, float]): The lower and upper bound of values. Default: (0, 1)
        annealing_limit (float): The proportion limit to stop annealing the value until
            next cycle. Default: 0.5
        track_values (bool): Track values produced during annealing. Default: False
    """

    def __init__(
        self,
        num_steps: int = 100,
        num_cycles: int = 4,
        bounds: tuple[float, float] = (0, 1),
        annealing_limit: float = 0.5,
        track_values: bool = False,
    ):
        super().__init__(track_values)
        self.num_steps = num_steps
        self.num_cycles = num_cycles
        self.min_val, self.max_val = bounds
        self.annealing_limit = annealing_limit

    @property
    def num_steps_per_cycle(self):
        """Number of steps per cycle"""
        return self.num_steps / self.num_cycles

    def get_current_value(self):
        tau = fmod(self._current_step, ceil(self.num_steps_per_cycle))
        tau /= self.num_steps_per_cycle
        f_tau = tau / self.annealing_limit

        if tau <= self.annealing_limit:
            return f_tau * (self.max_val - self.min_val) + self.min_val
        return self.max_val

    def extra_repr(self):
        if self._repr_mode == "state":
            return super().extra_repr()

        bounds = (self.min_val, self.max_val)
        return (
            f"num_steps={self.num_steps}, num_cycles={self.num_cycles}, "
            f"bounds={bounds}, annealing_limit={self.annealing_limit}, "
            f"track_values={self.track_values}"
        )


class AnnealerContainer(BaseAnnealer):
    """Container to store multiple value annealers.

    Args:
        **annealers (dict[str, BaseAnnealer]): Multiple annealers to store
    """

    def __init__(self, **annealers: dict[str, BaseAnnealer]):
        self.annealers = annealers
        super().__init__()
        self.get_mode()

    def __getitem__(self, keys: str | list[str] | slice) -> dict[str, float] | float:
        """Gets the value from the specified annealers. Use [:] to return all current values.

        Args:
            keys (str | list[str] | slice): The dictionary key to obtain the values.

        Returns:
            dict[str, float] | float: The current value(s) from the indexed annealers.

        Raises:
            ValueError: If slice is used but is not [:].
        """
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

    def step(self, keys: str | list[str] | None = None, increment: int = 1):
        """Increment the current step for specified or all annealers.

        Args:
            keys (str | list[str] | None): The dictionary key to increment the step. Default: None
            increment (int): Value for increment. Default: 1
        """
        super().step(increment)

        if keys is None:
            self.global_step()
            return

        if isinstance(keys, str):
            _check_missing_keys(keys, self.annealers)
            self.annealers[keys].step(increment)
            return

        for k in keys:
            _check_missing_keys(k, self.annealers)
            self.annealers[k].step(increment)

    def global_step(self, increment: int = 1):
        """Call step for all available annealers.

        Args:
            increment (int): Value for increment. Default: 1
        """
        for v in self.annealers.values():
            v.step(increment)

    def extra_repr(self):
        repr = pprint.pformat(self.annealers, width=1, indent=1).strip("{}")
        repr = textwrap.indent(repr, "    ")
        return repr

    def __repr__(self):
        return f"{self.__class__.__name__}({{\n{self.extra_repr()}\n}})"

    def get_mode(self, mode: str = "value"):
        """Set get mode to obtain either value or the annealer itself.

        Args:
            mode (str): Mode to obtain values, either "value" or "object". Default: "value"
        """
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
