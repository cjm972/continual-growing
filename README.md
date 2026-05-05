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

## Adding a new hyperparameter

Hyperparameters are defined in **one place**: the argparse parser in
`src/run.py`. Everything else (configs, HPO search, validation) keys off
that single declaration.

1. **Declare the flag** in `src/run.py`. Pick a unique `dest` — that
   name is what configs and `!hpo:*` tags will reference.

   ```python
   parser.add_argument('--dropout_p', default=0.0, type=float,
                       help='Dropout probability on hidden activations.')
   ```

   If the value is from a fixed set, add `choices=[...]` — config-loaded
   values are validated against it for free.

2. **Mirror the default** in `configs/defaults.yaml` so `--config
   defaults.yaml` produces the same args as a no-flag CLI run:

   ```yaml
   dropout_p: 0.0
   ```

3. **Use the value** wherever you need it (`args.dropout_p` in
   `runtime.py` / `train/trainer.py` / `networks/...`). The HPO runner
   constructs a fresh `args` namespace per trial via the same parser, so
   no extra plumbing is needed.

That's it for plain runs. To make the new arg **searchable**, add an
`!hpo:*` tag in any HPO config:

```yaml
dropout_p: !hpo:float 0.0, 0.5
```

The runner picks it up automatically — no allow-list, no registration
step. Optuna will see the parameter under its dotted path (`dropout_p`),
the search-space hash will include it, and it'll appear in `trials.csv`,
`best.yaml`, and the parameter-importance plots.

**Gotchas:**

- **Leaf names must be unique** across the merged YAML tree. If you put
  `training.dropout_p` in one config and `regularization.dropout_p` in
  another, the loader rejects the merge — pick one location and stick
  to it across files.
- **CLI flags always win.** Passing `--dropout_p 0.3` overrides any
  config value (and pins it as a constant during HPO if you launch HPO
  with that flag).
- **Type coercion is argparse's.** YAML scalars are passed through
  `argparse`-equivalent type coercion at apply time, so
  `dropout_p: "0.3"` becomes `0.3` (float).
- **Anything is searchable** — `device`, `data_path`, etc. included.
  No guardrails: if you tag `!hpo:choice cuda, cpu` on `device`, that's
  what you'll get. Use accordingly.

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
