"""HPO CLI: `python -m hpo.cli --experiment X --config foo.yaml [bar.yaml ...]`.

Composes the search space from layered configs, creates/attaches an Optuna
study, runs `n_trials`, writes artifacts. Ctrl+C stops cleanly: the
in-flight trial is recorded as PRUNED and the driver finalises outputs
from whatever completed trials exist.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from hpo.experiment import compose_search_config, search_space_hash
from hpo.outputs import finalize_study_outputs
from hpo.runner import make_objective
from hpo.study import (
    ALLOWED_AGGREGATES,
    default_storage,
    make_study,
    metric_directions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HPO_RUNS_DIR = REPO_ROOT / "hpo_runs"


def _ensure_cwd_is_src() -> None:
    """Plain `import utils, dataloaders, networks` only works from src/."""
    src = REPO_ROOT / "src"
    if Path.cwd().resolve() != src.resolve():
        os.chdir(src)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))


def _build_base_parser() -> argparse.ArgumentParser:
    """Mirror src/run.py's argparse — used to validate config keys and to
    construct trial args. Imported via runpy-ish lazy load to avoid
    triggering run.py's top-level execution."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='mps', type=str)
    parser.add_argument('--wandb_mode', default='online', type=str,
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--experiment', default='mnist5', type=str,
                        choices=['mnist2', 'mnist5', 'pmnist', 'cifar', 'mixture'])
    parser.add_argument('--train_mode', default='grow', type=str)
    parser.add_argument('--data_path', default='../data/', type=str)
    parser.add_argument('--cl_mode', default='domain-incremental', type=str,
                        choices=['task-incremental', 'domain-incremental'])
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--checkpoint_dir', default='../checkpoints/', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--sbatch', default=64, type=int)
    parser.add_argument('--lr_mu', default=0.01, type=float)
    parser.add_argument('--lr_sigma', default=0.01, type=float)
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--hidden_n', default=16, type=int)
    parser.add_argument('--parameter', default='', type=str)
    parser.add_argument('--samples', default=10, type=int)
    parser.add_argument('--sigma_init', default=0.1, type=float)
    parser.add_argument('--sigma_prior1', default=1.0, type=float)
    parser.add_argument('--sigma_prior2', default=0.00001, type=float)
    parser.add_argument('--pi', default=0.25, type=float)
    parser.add_argument('--rho_init_mode', default='gaussian', type=str,
                        choices=['gaussian', 'bimodal'])
    parser.add_argument('--regularization', default='sns', type=str,
                        choices=['bbb', 'sns', 'unimodal'])
    parser.add_argument('--arch', default='mlp', type=str)
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--growth_rate', default=5, type=int)
    parser.add_argument('--growth_saturation', default=0.2, type=float)
    parser.add_argument('--growth_threshold', default=0.05, type=float)
    parser.add_argument('--resume', default='no', type=str)
    parser.add_argument('--sti', default=0, type=int)
    parser.add_argument('--valid_split', default=0.0, type=float)
    parser.add_argument('--config', default=[], nargs='+')
    return parser


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="continual-growing-hpo",
        description="Optuna HPO for continual-growing.",
    )
    ap.add_argument('--config', '-c', nargs='+', required=True,
                    help='YAML config(s) under configs/ (last wins).')
    ap.add_argument('--experiment', type=str, default=None,
                    help='Override the experiment from config (one of mnist2/mnist5/pmnist/cifar/mixture).')
    ap.add_argument('--n-trials', type=int, default=None,
                    help='Override hpo.n_trials from the config.')
    ap.add_argument('--study-name', type=str, default=None,
                    help='Override hpo.study_name from the config.')
    args = ap.parse_args(argv)

    _ensure_cwd_is_src()

    merged, search_space = compose_search_config(args.config)
    if args.experiment is not None:
        merged["experiment"] = args.experiment

    hpo_block = merged.get("hpo") or {}
    if not isinstance(hpo_block, dict):
        raise ValueError(f"`hpo:` block must be a mapping, got {type(hpo_block).__name__}")

    metrics = list(hpo_block.get("metrics") or ["avg_acc"])
    n_trials = int(args.n_trials if args.n_trials is not None
                   else hpo_block.get("n_trials", 10))
    n_jobs = int(hpo_block.get("n_jobs", 1))
    if n_jobs != 1:
        raise NotImplementedError(
            "n_jobs>1 is not supported yet. For parallel HPO, run multiple "
            "`continual-growing-hpo` workers against a shared hpo.storage URL."
        )
    master_seed = int(hpo_block.get("master_seed", 0))
    n_seeds = int(hpo_block.get("n_seeds", 1))
    aggregate = str(hpo_block.get("aggregate", "mean"))
    if aggregate not in ALLOWED_AGGREGATES:
        raise ValueError(
            f"hpo.aggregate must be one of {sorted(ALLOWED_AGGREGATES)}, got {aggregate!r}"
        )
    valid_split = float(hpo_block.get("valid_split", 0.1))
    save_trial_artifacts = bool(hpo_block.get("save_trial_artifacts", False))
    save_checkpoints = bool(hpo_block.get("save_checkpoints", False))
    wandb_per_trial = bool(hpo_block.get("wandb_per_trial", True))

    # Pre-compute hash to feed default study_name and to gate resume.
    space_hash = search_space_hash(search_space)
    explicit_name = args.study_name if args.study_name is not None else hpo_block.get("study_name")
    if explicit_name:
        study_name = str(explicit_name)
    else:
        exp = str(merged.get("experiment", "study"))
        study_name = f"{exp}_{space_hash[:8]}"

    study_dir = HPO_RUNS_DIR / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    storage = hpo_block.get("storage")
    if not storage:
        storage = default_storage(study_name, study_dir)

    sampler_block = hpo_block.get("sampler") or ("nsga2" if len(metrics) > 1 else "tpe")
    if isinstance(sampler_block, str):
        sampler_name = sampler_block
        sampler_kwargs: dict = {}
    else:
        sampler_name = str(sampler_block.get("name", "tpe"))
        sampler_kwargs = dict(sampler_block.get("kwargs") or {})

    pruner_block = hpo_block.get("pruner") or "nop"
    if isinstance(pruner_block, str):
        pruner_name = pruner_block
        pruner_kwargs: dict = {}
    else:
        pruner_name = str(pruner_block.get("name", "nop"))
        pruner_kwargs = dict(pruner_block.get("kwargs") or {})

    wandb_mode = str(hpo_block.get("wandb_mode") or merged.get("wandb_mode") or "online")

    # Sanity-check direction inference.
    metric_directions(metrics)

    study = make_study(
        study_name=study_name,
        storage=storage,
        metrics=metrics,
        sampler_name=sampler_name,
        sampler_kwargs=sampler_kwargs,
        pruner_name=pruner_name,
        pruner_kwargs=pruner_kwargs,
        master_seed=master_seed,
        search_space_hash_value=space_hash,
    )

    base_parser = _build_base_parser()
    objective = make_objective(
        base_parser=base_parser,
        base_dict=merged,
        search_space=search_space,
        metrics=metrics,
        master_seed=master_seed,
        n_seeds=n_seeds,
        aggregate=aggregate,
        valid_split=valid_split,
        study_dir=study_dir,
        study_name=study_name,
        save_trial_artifacts=save_trial_artifacts,
        save_checkpoints=save_checkpoints,
        wandb_per_trial=wandb_per_trial,
        wandb_mode=wandb_mode,
    )

    print(f"[hpo] study_name={study_name}")
    print(f"[hpo] storage={storage}")
    print(f"[hpo] metrics={metrics}, n_trials={n_trials}, n_seeds={n_seeds}")
    print(f"[hpo] search_space ({len(search_space)} dims):")
    for path, spec in search_space:
        print(f"  - {path}: {spec}")

    # Ctrl+C → stop after current trial finishes; second Ctrl+C → hard abort.
    def _on_sigint(signum, frame):
        print("\n[hpo] Ctrl+C received — finishing current trial then stopping.")
        study.stop()
        signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGINT, _on_sigint)

    tie_break = hpo_block.get("tie_break")

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=())
    finally:
        finalize_study_outputs(
            study_dir, study, merged, search_space, metrics, tie_break,
        )
        print(f"[hpo] artifacts written to {study_dir}")

    if len(metrics) == 1 and study.best_trial is not None:
        bt = study.best_trial
        print(f"[hpo] best trial: #{bt.number}  value={bt.value:.6f}")
        print(f"[hpo] params: {bt.params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
