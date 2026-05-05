"""Hyperparameter optimisation for continual-growing.

Sub-modules:
- `tags`: YAML constructors for `!hpo:*` search-space tags.
- `space`: SearchSpec dataclasses + Optuna materialisation.
- `config`: layered --config loading (deep-merge, flatten, apply).
- `experiment`: search-space composition over the merged config.
- `study`: Optuna study factory (sampler, pruner, storage).
- `runner`: per-trial objective.
- `cli`: `python -m hpo.cli` entry point.
"""
