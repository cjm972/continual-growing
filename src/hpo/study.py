"""Optuna study factory: sampler, pruner, storage, resume gating.

The whole study is bit-reproducible from one integer (`master_seed`):
samplers are seeded from it, and per-trial seeds are derived deterministically
inside the runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

# Metric → Optuna direction. Single source of truth so the user can't
# pick a wrong-direction combo.
METRIC_DIRECTIONS = {
    "avg_acc": "maximize",
    "bwt": "maximize",
    "loss": "minimize",
}

ALLOWED_SAMPLERS = frozenset({"tpe", "cmaes", "random", "nsga2"})
ALLOWED_PRUNERS = frozenset({"nop", "median", "hyperband"})
ALLOWED_AGGREGATES = frozenset({"mean", "median", "trimmed_mean"})


def metric_directions(metrics: Sequence[str]) -> list[str]:
    out = []
    for m in metrics:
        if m not in METRIC_DIRECTIONS:
            raise ValueError(
                f"unknown metric {m!r}; allowed: {sorted(METRIC_DIRECTIONS)}"
            )
        out.append(METRIC_DIRECTIONS[m])
    return out


def build_sampler(name: str, kwargs: dict[str, Any], master_seed: int) -> Any:
    import optuna

    if name not in ALLOWED_SAMPLERS:
        raise ValueError(
            f"unknown sampler {name!r}; allowed: {sorted(ALLOWED_SAMPLERS)}"
        )
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=master_seed, **kwargs)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=master_seed, **kwargs)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=master_seed, **kwargs)
    if name == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=master_seed, **kwargs)
    raise ValueError(name)  # unreachable


def build_pruner(name: str, kwargs: dict[str, Any]) -> Any:
    import optuna

    if name not in ALLOWED_PRUNERS:
        raise ValueError(
            f"unknown pruner {name!r}; allowed: {sorted(ALLOWED_PRUNERS)}"
        )
    if name == "nop":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(**kwargs)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(**kwargs)
    raise ValueError(name)  # unreachable


def default_storage(study_name: str, study_dir: Path) -> str:
    """Default sqlite storage URL (study DB lives inside the study dir)."""
    study_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{study_dir}/study.db"


def make_study(
    *,
    study_name: str,
    storage: str,
    metrics: Sequence[str],
    sampler_name: str,
    sampler_kwargs: dict[str, Any],
    pruner_name: str,
    pruner_kwargs: dict[str, Any],
    master_seed: int,
    search_space_hash_value: str,
) -> Any:
    """Create or attach to a study, gated by `search_space_hash`."""
    import optuna

    directions = metric_directions(metrics)
    sampler = build_sampler(sampler_name, sampler_kwargs, master_seed)
    pruner = build_pruner(pruner_name, pruner_kwargs)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        directions=directions,
        load_if_exists=True,
    )

    existing_hash = study.user_attrs.get("search_space_hash")
    if existing_hash is None:
        study.set_user_attr("search_space_hash", search_space_hash_value)
        study.set_user_attr("metrics", list(metrics))
    elif existing_hash != search_space_hash_value:
        raise RuntimeError(
            f"search space changed for study {study_name!r} "
            f"(stored hash {existing_hash} != new hash {search_space_hash_value}); "
            "revert the YAML or pick a new study_name"
        )
    return study
