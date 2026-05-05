"""Per-study artifact writers: experiment.yaml, search_space.yaml,
trials.csv, best.yaml / best_<metric>.yaml / pareto.csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from hpo.experiment import (
    search_space_to_jsonable,
    splice_value,
    stripped_concrete_dict,
)
from hpo.space import SearchSpec


def write_experiment_yaml(study_dir: Path, merged: dict[str, Any]) -> None:
    """Dump the merged HPO config with specs replaced by `null` placeholders."""
    out = stripped_concrete_dict(merged)
    (study_dir / "experiment.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False)
    )


def write_search_space_yaml(
    study_dir: Path, search_space: list[tuple[str, SearchSpec]]
) -> None:
    payload = search_space_to_jsonable(search_space)
    (study_dir / "search_space.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )


def write_trials_csv(study_dir: Path, study: Any, metrics: Sequence[str]) -> None:
    """Write `trials.csv` with `trial_number, state, value(s), params...`."""
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    param_keys = sorted({k for t in trials for k in (t.params or {}).keys()})
    with (study_dir / "trials.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["trial_number", "state"]
        if len(metrics) == 1:
            header.append(f"value_{metrics[0]}")
        else:
            header.extend(f"value_{m}" for m in metrics)
        header.extend(param_keys)
        writer.writerow(header)
        for t in trials:
            row: list[Any] = [t.number, t.state.name]
            if t.values is None:
                row.extend([""] * len(metrics))
            else:
                row.extend(t.values)
            for k in param_keys:
                row.append((t.params or {}).get(k, ""))
            writer.writerow(row)


def _concrete_config_from_trial(
    base_dict: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Splice a trial's params into a deep-copied merged dict, strip residuals."""
    import copy
    out = copy.deepcopy(base_dict)
    for path, value in params.items():
        splice_value(out, path, value)
    return stripped_concrete_dict(out)


def write_best_yaml(
    study_dir: Path,
    base_dict: dict[str, Any],
    trial: Any,
    name: str = "best.yaml",
) -> None:
    config = _concrete_config_from_trial(base_dict, trial.params or {})
    config.pop("hpo", None)
    (study_dir / name).write_text(yaml.safe_dump(config, sort_keys=False))


def write_pareto_csv(
    study_dir: Path, study: Any, metrics: Sequence[str]
) -> None:
    """Write the Pareto-optimal trials of a multi-objective study."""
    pareto = study.best_trials
    param_keys = sorted({k for t in pareto for k in (t.params or {}).keys()})
    with (study_dir / "pareto.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["trial_number"] + [f"value_{m}" for m in metrics] + param_keys
        writer.writerow(header)
        for t in pareto:
            row: list[Any] = [t.number]
            row.extend(t.values or [])
            for k in param_keys:
                row.append((t.params or {}).get(k, ""))
            writer.writerow(row)


def select_corner_trials(
    study: Any, metrics: Sequence[str]
) -> dict[str, Any]:
    """Pick the trial that maxes each individual objective (after applying
    the metric's direction sign)."""
    from hpo.study import metric_directions
    directions = metric_directions(metrics)
    completed = [t for t in study.get_trials(deepcopy=False)
                 if t.values is not None]
    out: dict[str, Any] = {}
    for i, m in enumerate(metrics):
        if not completed:
            continue
        sign = 1.0 if directions[i] == "maximize" else -1.0
        best = max(completed, key=lambda t: sign * t.values[i])
        out[m] = best
    return out


def reduce_pareto_with_tie_break(
    study: Any, metrics: Sequence[str], tie_break: dict[str, Any]
) -> Any | None:
    """Pick a single trial from the Pareto front via weights or metric index."""
    pareto = study.best_trials
    if not pareto:
        return None
    weights = tie_break.get("weights")
    metric_idx = tie_break.get("metric")
    if (weights is None) == (metric_idx is None):
        raise ValueError(
            "tie_break must specify exactly one of `weights` or `metric`"
        )
    if weights is not None:
        weights = [float(w) for w in weights]
        if len(weights) != len(metrics):
            raise ValueError(
                f"tie_break.weights length {len(weights)} != metrics {len(metrics)}"
            )
        return max(pareto, key=lambda t: sum(w * v for w, v in zip(weights, t.values)))
    return max(pareto, key=lambda t: t.values[int(metric_idx)])


def write_pareto_trial_dirs(
    study_dir: Path,
    study: Any,
    base_dict: dict[str, Any],
    metrics: Sequence[str],
) -> None:
    """For each trial on the Pareto front, write `pareto/trial-<n>/` with:
      - `config.yaml` — concrete config, re-runnable as `--config`
      - `metrics.json` — values + params
      - CL plots from the trial's acc_matrix (when available)
    """
    import json
    from hpo.plots import write_cl_plots_for_trial

    pareto = study.best_trials
    base = study_dir / "pareto"
    base.mkdir(parents=True, exist_ok=True)
    for t in pareto:
        trial_dir = base / f"trial-{t.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = _concrete_config_from_trial(base_dict, t.params or {})
        config.pop("hpo", None)
        (trial_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (trial_dir / "metrics.json").write_text(
            json.dumps(
                {"trial_number": t.number,
                 "metrics": dict(zip(metrics, list(t.values or []))),
                 "params": dict(t.params or {})},
                indent=2,
            )
        )
        write_cl_plots_for_trial(
            t, trial_dir, label=f"pareto trial #{t.number}",
        )


def finalize_study_outputs(
    study_dir: Path,
    study: Any,
    merged_dict: dict[str, Any],
    search_space: list[tuple[str, SearchSpec]],
    metrics: list[str],
    tie_break: dict[str, Any] | None = None,
) -> None:
    """Write experiment.yaml, search_space.yaml, trials.csv, best*.yaml,
    plots/, and (multi-obj) pareto/trial-<n>/ dirs."""
    from hpo.plots import make_study_plots

    write_experiment_yaml(study_dir, merged_dict)
    write_search_space_yaml(study_dir, search_space)
    write_trials_csv(study_dir, study, metrics)

    completed = [t for t in study.get_trials(deepcopy=False)
                 if t.values is not None]
    if not completed:
        print("[outputs] no completed trials; skipping best/pareto/plots")
        return

    if len(metrics) == 1:
        best = study.best_trial
        write_best_yaml(study_dir, merged_dict, best, name="best.yaml")
        make_study_plots(study_dir, study, metrics)
        return

    # Multi-objective.
    write_pareto_csv(study_dir, study, metrics)
    corners = select_corner_trials(study, metrics)
    for m, t in corners.items():
        write_best_yaml(study_dir, merged_dict, t, name=f"best_{m}.yaml")
    if tie_break:
        chosen = reduce_pareto_with_tie_break(study, metrics, tie_break)
        if chosen is not None:
            write_best_yaml(study_dir, merged_dict, chosen, name="best.yaml")
    write_pareto_trial_dirs(study_dir, study, merged_dict, metrics)
    make_study_plots(study_dir, study, metrics)
