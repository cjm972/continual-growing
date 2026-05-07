"""Single source of truth for run/HPO hyperparameters.

`Params` lists every flag, its default, type, choices, and help text in
one place. `build_parser()` materialises a matching argparse parser used
by both `run.py` and the HPO CLI — no more drift between two parser
declarations.

Adding a new hyperparameter is a one-line change here, picked up
automatically by every caller (CLI, configs, HPO trials).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


def _flag(
    default: Any,
    *,
    type: Any = None,
    help: str = "",
    choices: list | None = None,
    action: str | None = None,
    nargs: str | int | None = None,
    required: bool = False,
):
    """Build a dataclass field with argparse metadata attached.

    Mutable defaults (lists) are wrapped in a default_factory so each
    `Params()` gets its own copy.
    """
    md: dict[str, Any] = {
        "type": type,
        "help": help,
        "choices": choices,
        "action": action,
        "nargs": nargs,
        "required": required,
    }
    if isinstance(default, list):
        return field(default_factory=lambda d=default: list(d), metadata=md)
    return field(default=default, metadata=md)


@dataclass
class Params:
    # --- Core run setup ---------------------------------------------------
    seed: int = _flag(0, type=int, help="Random seed.")
    device: str = _flag("mps", type=str, help="torch device (cpu / cuda / mps).")
    wandb_mode: str = _flag(
        "online", type=str,
        choices=["online", "offline", "disabled"],
        help="Wandb backend mode.",
    )
    experiment: str = _flag(
        "mnist5", type=str, required=True,
        choices=["mnist2", "mnist5", "pmnist", "cifar", "mixture"],
        help="Continual-learning benchmark to run.",
    )
    train_mode: str = _flag(
        "grow", type=str, help="Training regime (e.g. 'grow', 'joint').",
    )
    data_path: str = _flag("../data/", type=str, help="Dataset root.")
    cl_mode: str = _flag(
        "domain-incremental", type=str,
        choices=["task-incremental", "domain-incremental"],
        help="Continual-learning mode.",
    )

    # --- Training ---------------------------------------------------------
    output: str = _flag("", type=str, help="Output subdir name (auto if empty).")
    checkpoint_dir: str = _flag("../checkpoints/", type=str, help="Checkpoint root.")
    epochs: int = _flag(10, type=int, help="Epochs per task.")
    sbatch: int = _flag(64, type=int, help="Batch size.")
    lr_mu: float = _flag(0.01, type=float, help="Learning rate for means (mu).")
    lr_sigma: float = _flag(0.01, type=float, help="Learning rate for uncertainties (rho).")
    layers: int = _flag(1, type=int, help="Number of hidden layers.")
    hidden_n: int = _flag(16, type=int, help="Hidden width.")
    parameter: str = _flag("", type=str, help="Free-form parameter slot.")

    # --- Bayesian ---------------------------------------------------------
    samples: int = _flag(10, type=int, help="Monte Carlo samples per forward.")
    sigma_init: float = _flag(0.1, type=float, help="Initial posterior stdev.")
    sigma_prior1: float = _flag(1.0, type=float, help="Stdev of 1st prior component.")
    sigma_prior2: float = _flag(0.00001, type=float, help="Stdev of 2nd prior component.")
    pi: float = _flag(0.25, type=float, help="Prior mixture weight.")
    rho_init_mode: str = _flag(
        "gaussian", type=str, choices=["gaussian", "bimodal"],
        help="Initialisation mode for rho.",
    )
    regularization: str = _flag(
        "sns", type=str, choices=["bbb", "sns", "unimodal"],
        help="Regularisation method.",
    )
    arch: str = _flag("mlp", type=str, help="Network architecture.")
    static: bool = _flag(
        False, action="store_true",
        help="Use only means (no Bayesian sampling).",
    )

    # --- Growth -----------------------------------------------------------
    growth_rate: int = _flag(5, type=int, help="Neurons added per growth step (0 disables growth).")
    growth_saturation: float = _flag(0.2, type=float, help="Saturation fraction trigger.")
    growth_threshold: float = _flag(0.05, type=float, help="Stdev threshold for saturation.")

    # --- Lateral inhibition ----------------------------------------------
    successive_inhibition: bool = _flag(
        False, action="store_true",
        help="Enable asymmetric successive lateral inhibition.",
    )
    inhibition_samples: int = _flag(
        5, type=int,
        help="Samples used to estimate pre-activation variance for inhibition.",
    )
    gamma_inhibition: float = _flag(
        0.5, type=float, help="Scaling factor for inhibition strength.",
    )

    # --- Resume / continuation -------------------------------------------
    resume: str = _flag("no", type=str, help="Resume from checkpoint? ('yes' / 'no').")
    sti: int = _flag(0, type=int, help="Starting task index.")
    valid_split: float = _flag(
        0.0, type=float,
        help="Per-task valid holdout fraction. 0.0 = legacy valid==train.",
    )


def build_parser(
    *,
    cls: type = Params,
    add_config: bool = True,
    description: str = "",
    enforce_required: bool = True,
) -> argparse.ArgumentParser:
    """Materialise an argparse parser from a dataclass schema.

    `add_config=True` appends the `--config` flag (plumbing for layered
    YAML loads, not part of the model schema).

    `enforce_required=False` drops the `required=True` constraint on
    every flag — used by the HPO runner when building trial args from a
    config (the YAML supplies the value, no CLI parse happens).
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Concrete defaults live in configs/defaults.yaml.",
    )
    for f in fields(cls):
        kwargs: dict[str, Any] = {"help": f.metadata.get("help", "")}
        action = f.metadata.get("action")
        if action:
            kwargs["action"] = action
            # store_true / store_false handle their own default + type.
            if action in ("store_true", "store_false"):
                kwargs["default"] = (action == "store_false")
            else:
                kwargs["default"] = _resolve_default(f)
        else:
            kwargs["default"] = _resolve_default(f)
            t = f.metadata.get("type")
            if t is not None:
                kwargs["type"] = t
            choices = f.metadata.get("choices")
            if choices is not None:
                kwargs["choices"] = choices
            nargs = f.metadata.get("nargs")
            if nargs is not None:
                kwargs["nargs"] = nargs
        if enforce_required and f.metadata.get("required"):
            kwargs["required"] = True
        parser.add_argument(f"--{f.name}", **kwargs)

    if add_config:
        parser.add_argument(
            "--config", default=[], nargs="+",
            help="YAML config file(s) under configs/. Layered left-to-right "
                 "(last wins). CLI flags win over configs.",
        )
    return parser


def _resolve_default(f) -> Any:
    """Pull the actual default value from a dataclass field (handles factories)."""
    if f.default is not _MISSING:
        return f.default
    if f.default_factory is not _MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    raise ValueError(f"field {f.name!r} has no default")


# dataclasses.MISSING is the sentinel for "no default"; aliased once so the
# helper is concise.
from dataclasses import MISSING as _MISSING  # noqa: E402


def params_to_dict(cls: type = Params) -> dict[str, Any]:
    """Default Params() as a flat {name: value} dict, suitable for
    regenerating configs/defaults.yaml."""
    p = cls()
    return {f.name: getattr(p, f.name) for f in fields(cls)}
