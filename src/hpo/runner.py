"""Per-trial objective: materialise → splice → run training → reduce → report.

Trial seed is derived deterministically from `(master_seed, trial.number,
seed_idx)` via SHA-256 so `master_seed` reproduces the entire study.

Pruning: per-task `trial.report(running_avg_acc, step=t)` then
`trial.should_prune()`. Disabled silently for multi-objective studies
(Optuna refuses report() in that case).

NaN truncation: `trainer.py` already breaks the epoch loop when valid_loss
becomes NaN and restores `best_model`. We compute the trial metric on the
restored model, so diverged trials return their pre-divergence best metric
rather than NaN.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from hpo.experiment import splice_value
from hpo.space import (
    CategoricalSpec,
    FloatSpec,
    IntSpec,
    SearchSpec,
    materialise,
)

_UINT32_MASK = (1 << 32) - 1


def stable_hash(args: tuple[Any, ...]) -> int:
    """SHA-256 first 4 bytes → uint32. 32-bit because numpy.random.seed
    rejects anything wider; torch/cuda accept 32-bit fine."""
    payload = repr(args).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return struct.unpack(">I", digest[:4])[0] & _UINT32_MASK


def trial_seed(master_seed: int, trial_number: int, seed_idx: int = 0) -> int:
    return stable_hash(("continual-growing", int(master_seed), int(trial_number), int(seed_idx)))


def _materialise_search_space(
    trial: Any, search_space: list[tuple[str, SearchSpec]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, spec in search_space:
        out[path] = materialise(spec, trial, path)
    return out


def _strip_residual_specs(node: Any) -> Any:
    if isinstance(node, (FloatSpec, IntSpec, CategoricalSpec)):
        return None
    if isinstance(node, dict):
        return {k: _strip_residual_specs(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_residual_specs(v) for v in node]
    return node


def _build_trial_args(
    base_parser: argparse.ArgumentParser,
    base_dict: dict[str, Any],
    suggestions: dict[str, Any],
    *,
    seed: int,
    checkpoint_dir: Path,
) -> argparse.Namespace:
    """Merge base config + suggestions into a fresh args namespace."""
    from hpo.config import flatten_to_argparse

    resolved = copy.deepcopy(base_dict)
    for path, value in suggestions.items():
        splice_value(resolved, path, value)
    resolved = _strip_residual_specs(resolved)
    flat, _ = flatten_to_argparse(resolved, base_parser)

    args = base_parser.parse_args([])  # all defaults
    for k, v in flat.items():
        setattr(args, k, v)
    args.seed = seed
    args.checkpoint_dir = str(checkpoint_dir)
    args.output = ""  # let runtime.run_training derive it
    args.config = []
    return args


def _compute_metric(metric: str, avg_acc: float, bwt: float) -> float:
    if metric == "avg_acc":
        return float(avg_acc)
    if metric == "bwt":
        return float(bwt)
    if metric == "loss":
        # No held-out aggregate "loss" — proxy with negative avg_acc so
        # lower is better. (Ad-hoc; prefer avg_acc/bwt for now.)
        return float(-avg_acc)
    raise ValueError(f"unknown metric {metric!r}")


def _aggregate(values: list[float], rule: str) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if rule == "mean":
        return float(arr.mean())
    if rule == "median":
        return float(np.median(arr))
    if rule == "trimmed_mean":
        n = arr.shape[0]
        k = max(1, n // 10)
        if n - 2 * k <= 0:
            return float(arr.mean())
        return float(np.sort(arr)[k : n - k].mean())
    raise ValueError(f"unknown aggregate {rule!r}")


def make_objective(
    *,
    base_parser: argparse.ArgumentParser,
    base_dict: dict[str, Any],
    search_space: list[tuple[str, SearchSpec]],
    metrics: list[str],
    master_seed: int,
    n_seeds: int,
    aggregate: str,
    valid_split: float,
    study_dir: Path,
    study_name: str,
    save_trial_artifacts: bool,
    save_checkpoints: bool,
    wandb_per_trial: bool,
    wandb_mode: str,
):
    """Closure: returns an `objective(trial)` callable for `study.optimize`."""

    import optuna
    import wandb

    n_objectives = len(metrics)

    def objective(trial: Any) -> float | tuple[float, ...]:
        suggestions = _materialise_search_space(trial, search_space)
        # When save_trial_artifacts=False (default) the trial writes into a
        # tempdir we nuke in the finally block — nothing per-trial lands on
        # disk. Trial state lives in study.db + trials.csv either way.
        if save_trial_artifacts:
            trial_dir = study_dir / "trials" / f"{trial.number}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            cleanup_trial_dir = False
        else:
            trial_dir = Path(tempfile.mkdtemp(prefix=f"cg-hpo-trial-{trial.number}-"))
            cleanup_trial_dir = True

        per_seed_metrics: list[list[float]] = [[] for _ in metrics]

        for seed_idx in range(n_seeds):
            seed = trial_seed(master_seed, trial.number, seed_idx)
            args = _build_trial_args(
                base_parser, base_dict, suggestions,
                seed=seed, checkpoint_dir=trial_dir,
            )
            # Ensure HPO-controlled fields override anything from the config.
            args.valid_split = valid_split
            args.wandb_mode = wandb_mode

            # trainer.py calls wandb.log unconditionally, so wandb.init must
            # have been called either way. When wandb_per_trial=False we use
            # mode='disabled' to make all log calls no-op without changing
            # the trainer.
            effective_mode = wandb_mode if wandb_per_trial else "disabled"
            if effective_mode == "disabled":
                os.environ["WANDB_MODE"] = "disabled"
            wandb_run = wandb.init(
                project="continual_growing",
                group=study_name,
                name=f"trial_{trial.number}_seed_{seed_idx}",
                config={"trial_number": trial.number, "seed_idx": seed_idx, **suggestions},
                mode=effective_mode,
                reinit=True,
            )

            try:
                from runtime import run_training

                # Per-task pruning (single-objective only; Optuna refuses
                # trial.report on multi-obj). For multi-seed, the step axis
                # is packed so reports along the trial stay monotone.
                def _on_task(t: int, running_avg: float) -> None:
                    if n_objectives != 1:
                        return
                    step = seed_idx * 1000 + t  # ample buffer per seed
                    trial.report(float(running_avg), step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                avg_acc, bwt, _acc, _lss = run_training(args, on_task_done=_on_task)
                for i, m in enumerate(metrics):
                    per_seed_metrics[i].append(_compute_metric(m, avg_acc, bwt))

            except KeyboardInterrupt:
                # Honest signal — incomplete info, not a broken trial.
                raise optuna.TrialPruned()
            finally:
                if wandb_run is not None:
                    wandb_run.finish()
                if save_trial_artifacts and not save_checkpoints:
                    # Inside the kept trial dir, drop only the heavy
                    # per-task model snapshots; keep the .txt acc matrix
                    # and the final pickle for offline inspection.
                    for f in trial_dir.glob("model_*.pth.tar"):
                        try:
                            f.unlink()
                        except OSError:
                            pass

        finals = [_aggregate(buf, aggregate) for buf in per_seed_metrics]

        if save_trial_artifacts:
            import json
            import yaml as _yaml
            _yaml.safe_dump(
                {"trial_number": trial.number, "params": dict(suggestions)},
                (trial_dir / "params.yaml").open("w"), sort_keys=False,
            )
            (trial_dir / "metrics.json").write_text(
                json.dumps(
                    {"metrics": dict(zip(metrics, finals)),
                     "n_seeds": n_seeds, "aggregate": aggregate},
                    indent=2,
                )
            )
        elif cleanup_trial_dir:
            shutil.rmtree(trial_dir, ignore_errors=True)

        if n_objectives == 1:
            return finals[0]
        return tuple(finals)

    return objective
