"""Optuna + CL plots for a finished study.

Two families:
- Optuna's classical visualisations (matplotlib backend) — history,
  param importances, parallel coords, slice, contour, intermediates,
  Pareto front. Skipped per-plot if Optuna refuses (e.g. importances
  need ≥ 2 completed trials with varied params).
- Continual-learning plots from each best/Pareto trial's `acc_matrix`
  user_attr — heatmap of `acc[t, u]` and per-task forgetting curves.

Failures are isolated per plot so one missing dependency / degenerate
study doesn't kill the whole `finalize` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _safe(name: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — defensive for plot-only paths
        print(f"[plots] skipping {name}: {type(exc).__name__}: {exc}")


def _save_optuna_fig(fig_or_ax, path: Path) -> None:
    """Optuna's matplotlib viz returns Figure, Axes, or ndarray of Axes — cope."""
    import numpy as np
    obj = fig_or_ax
    if isinstance(obj, np.ndarray):
        obj = obj.flat[0]
    fig = getattr(obj, "figure", obj)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_optuna_single_obj(study: Any, plots_dir: Path) -> None:
    from optuna.visualization import matplotlib as ovm

    _safe("optimization_history", lambda: _save_optuna_fig(
        ovm.plot_optimization_history(study),
        plots_dir / "optimization_history.png",
    ))
    _safe("param_importances", lambda: _save_optuna_fig(
        ovm.plot_param_importances(study),
        plots_dir / "param_importances.png",
    ))
    _safe("parallel_coordinate", lambda: _save_optuna_fig(
        ovm.plot_parallel_coordinate(study),
        plots_dir / "parallel_coordinate.png",
    ))
    _safe("slice", lambda: _save_optuna_fig(
        ovm.plot_slice(study),
        plots_dir / "slice.png",
    ))
    if len(study.best_trial.params) >= 2:
        _safe("contour", lambda: _save_optuna_fig(
            ovm.plot_contour(study),
            plots_dir / "contour.png",
        ))
    _safe("intermediate_values", lambda: _save_optuna_fig(
        ovm.plot_intermediate_values(study),
        plots_dir / "intermediate_values.png",
    ))


def _plot_optuna_multi_obj(
    study: Any, metrics: Sequence[str], plots_dir: Path
) -> None:
    from optuna.visualization import matplotlib as ovm

    if len(metrics) == 2:
        _safe("pareto_front", lambda: _save_optuna_fig(
            ovm.plot_pareto_front(
                study, target_names=list(metrics),
            ),
            plots_dir / "pareto_front.png",
        ))
    # Per-target importances/parallel coords (one set per objective).
    for i, m in enumerate(metrics):
        _safe(f"param_importances_{m}", lambda i=i, m=m: _save_optuna_fig(
            ovm.plot_param_importances(
                study, target=lambda t, i=i: t.values[i], target_name=m,
            ),
            plots_dir / f"param_importances_{m}.png",
        ))
        _safe(f"parallel_coordinate_{m}", lambda i=i, m=m: _save_optuna_fig(
            ovm.plot_parallel_coordinate(
                study, target=lambda t, i=i: t.values[i], target_name=m,
            ),
            plots_dir / f"parallel_coordinate_{m}.png",
        ))


def _plot_acc_matrix(acc: Any, out_path: Path, title: str) -> None:
    """Heatmap of `acc[t, u]` (rows = task t just trained, cols = task u tested)."""
    import matplotlib.pyplot as plt
    import numpy as np

    acc = np.asarray(acc, dtype=float)
    n = acc.shape[0]
    fig, ax = plt.subplots(figsize=(0.6 * n + 2.5, 0.6 * n + 2.0))
    im = ax.imshow(acc, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("test task u")
    ax.set_ylabel("after training task t")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = acc[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v < 0.5 else "black"), fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_task_curve(acc: Any, out_path: Path, title: str) -> None:
    """One line per task u, x = training step t (after task t trained), y = acc[t, u].

    The drop after the diagonal is the forgetting signal.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    acc = np.asarray(acc, dtype=float)
    n = acc.shape[0]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xs = list(range(n))
    for u in range(n):
        ys = [acc[t, u] if t >= u else float("nan") for t in xs]
        ax.plot(xs, ys, marker="o", label=f"task {u}")
    ax.set_xticks(xs)
    ax.set_xlabel("after training task t")
    ax.set_ylabel("test accuracy on task u")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_cl_plots_for_trial(
    trial: Any, out_dir: Path, *, label: str
) -> None:
    """Heatmap + per-task curve from a trial's `acc_matrix` user_attr."""
    acc = trial.user_attrs.get("acc_matrix")
    if acc is None:
        print(f"[plots] trial #{trial.number}: no acc_matrix in user_attrs; "
              "skipping CL plots (was the trial PRUNED?)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe("cl_acc_matrix", _plot_acc_matrix,
          acc, out_dir / "cl_acc_matrix.png",
          title=f"{label} — acc[t, u]")
    _safe("cl_per_task_curve", _plot_per_task_curve,
          acc, out_dir / "cl_per_task_curve.png",
          title=f"{label} — per-task forgetting")


def make_study_plots(
    study_dir: Path,
    study: Any,
    metrics: Sequence[str],
) -> None:
    """Write Optuna's classical plots + best-trial CL plots to plots/."""
    plots_dir = study_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if len(metrics) == 1:
        _plot_optuna_single_obj(study, plots_dir)
        if study.best_trial is not None:
            write_cl_plots_for_trial(
                study.best_trial, plots_dir,
                label=f"best trial #{study.best_trial.number}",
            )
    else:
        _plot_optuna_multi_obj(study, metrics, plots_dir)
