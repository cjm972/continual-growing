# continual-growing

Continual learning with growing Bayesian neural networks.

## Install

```bash
# venv
python3 -m venv .venv && source .venv/bin/activate

# CUDA 12.8 wheel for torch (matches NVIDIA driver R550+; see pyproject.toml)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# project (optuna, matplotlib, fvcore are core deps)
pip install -e .
```

For uv users, `[tool.uv.sources]` in `pyproject.toml` selects the cu128 index
automatically — `uv pip install -e .` is enough.

CPU-only or different CUDA version: pass `--index-url` to point at the
matching pytorch wheel index (`cpu`, `cu118`, `cu124`, etc.) before the
project install.

## Plain run

```bash
cd src
python run.py --experiment mnist5 \
              --config defaults.yaml mnist5_grow.yaml
```

`--config` accepts one or more YAML files under `configs/` (short names
without `.yaml` are resolved against that directory). Files are
deep-merged left-to-right (last wins). Any explicit CLI flag wins over
configs:

```bash
python run.py --experiment mnist5 --config defaults.yaml --epochs 20
```

## HPO run

```bash
cd src
python -m hpo.cli --config hpo_lr_search.yaml
```

The HPO CLI loads layered configs (with `!hpo:*` tags resolved), composes
the search space, runs `hpo.n_trials` trials, and writes per-study
artifacts under `hpo_runs/<study_name>/`:

- `experiment.yaml` — the merged HPO config (specs nullified)
- `search_space.yaml` — JSON-able dump of the search space
- `study.db` — Optuna sqlite store
- `trials.csv` — rolling `trial_number, state, value(s), params...`
- `best.yaml` — concrete config for the best trial (re-runnable as `--config`)
- `pareto.csv` + `best_<metric>.yaml` — for multi-objective studies
- `pareto/trial-<n>/` — multi-objective only, one dir per Pareto-front
  trial: `config.yaml` (re-runnable), `metrics.json`, plus CL plots
  (`cl_acc_matrix.png`, `cl_per_task_curve.png`) for deeper analysis
- `plots/` — Optuna's classical visualisations (history, importances,
  parallel coords, slice, contour, intermediates for single-obj;
  Pareto front + per-objective importances/parallel coords for
  multi-obj) plus the best-trial CL plots in single-obj studies
- `trials/<n>/...` — only when `hpo.save_trial_artifacts: true`. Contains
  `params.yaml`, `metrics.json`, the per-task `.txt` accuracy matrix, and
  the final `<exp>_<mode>_seed_<seed>.p` pickle. Default is off — trial
  state is already in `study.db` and `trials.csv`, so writing a dir per
  trial is just clutter.

## YAML tag grammar

`!hpo:*` tags only resolve when loaded by `HPOLoader` (i.e. via the HPO
CLI). Plain runs must use untagged YAML.

| Tag              | Scalar form                                   | Spec produced                                       |
|------------------|-----------------------------------------------|-----------------------------------------------------|
| `!hpo:float`     | `low, high [, log=true] [, step=...]`         | `FloatSpec(low, high, log, step)`                   |
| `!hpo:int`       | `low, high [, log=true] [, step=...]`         | `IntSpec(low, high, log, step)`                     |
| `!hpo:loguniform`| `low, high`                                   | `FloatSpec(low, high, log=True)`                    |
| `!hpo:choice`    | `a, b, c, ...`                                | `CategoricalSpec(choices=(a, b, c, ...))`           |

Positional args first, kwargs last. Bare identifiers stay as strings.

Example:

```yaml
lr_mu:    !hpo:loguniform 1.0e-4, 1.0e-1
pi:       !hpo:float      0.05, 0.5
layers:   !hpo:int        1, 3
sampler:  !hpo:choice     tpe, cmaes, random
```

## Config schema

Top-level keys mirror the argparse flags in `src/run.py`. Nested grouping
is allowed but every leaf name must be unique across the merged tree —
the loader flattens by leaf-name match. Unknown leaf names error out.

The `hpo:` block is HPO-only. Recognised keys:

| Key                  | Default      | Notes                                                     |
|----------------------|--------------|-----------------------------------------------------------|
| `n_trials`           | 10           |                                                           |
| `n_jobs`             | 1            | only `1` supported in v1                                  |
| `master_seed`        | 0            | seeds samplers + per-trial seed derivation                |
| `n_seeds`            | 1            | replicas per trial; aggregate via `aggregate`             |
| `aggregate`          | `mean`       | `mean` / `median` / `trimmed_mean`                        |
| `metrics`            | `[avg_acc]`  | always a list; allowed: `avg_acc`, `bwt`, `loss`          |
| `sampler`            | `tpe`        | `tpe` / `cmaes` / `random` / `nsga2`                      |
| `pruner`             | `nop`        | `nop` / `median` / `hyperband`                            |
| `valid_split`        | `0.1`        | per-task valid holdout fraction                           |
| `wandb_per_trial`    | `true`       | one wandb run per trial, grouped by `study_name`          |
| `wandb_mode`         | inherits CLI | `online` / `offline` / `disabled`                         |
| `save_trial_artifacts` | `false`    | keep `hpo_runs/<study>/trials/<n>/` (otherwise tempdir nuked) |
| `save_checkpoints`   | `false`      | only meaningful with `save_trial_artifacts: true`; keeps `model_*.pth.tar` |
| `study_name`         | auto         | default: `<experiment>_<search_space_hash[:8]>`           |
| `storage`            | sqlite       | default `sqlite:///hpo_runs/<study_name>/study.db`        |
| `tie_break`          | `null`       | multi-obj: `weights: [...]` or `metric: <idx>`            |
