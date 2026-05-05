"""Search-space dataclasses + Optuna materialisation.

`materialise(spec, trial, name)` lazily imports Optuna and dispatches to
the corresponding `trial.suggest_*`. The dotted path that points at a spec
in the config tree is the Optuna parameter name — this is how the runner
stitches the suggested value back into the merged config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class FloatSpec:
    low: float
    high: float
    log: bool = False
    step: float | None = None


@dataclass(frozen=True)
class IntSpec:
    low: int
    high: int
    log: bool = False
    step: int = 1


@dataclass(frozen=True)
class CategoricalSpec:
    choices: tuple[Any, ...]


SearchSpec = Union[FloatSpec, IntSpec, CategoricalSpec]


def materialise(spec: SearchSpec, trial: Any, name: str) -> Any:
    """Call the matching `trial.suggest_*` for a `SearchSpec`."""
    if isinstance(spec, FloatSpec):
        return trial.suggest_float(
            name, spec.low, spec.high, log=spec.log, step=spec.step,
        )
    if isinstance(spec, IntSpec):
        return trial.suggest_int(
            name, spec.low, spec.high, log=spec.log, step=spec.step,
        )
    if isinstance(spec, CategoricalSpec):
        return trial.suggest_categorical(name, list(spec.choices))
    raise TypeError(f"unknown SearchSpec subtype: {type(spec).__name__}")
