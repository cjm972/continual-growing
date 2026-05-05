# continual-growing

Continual learning with Bayesian neural networks that **grow** their hidden layers when posterior uncertainty saturates. The growth signal comes from the variational posterior: when too many of the most-recently-added parameters' standard deviations have collapsed below a threshold, more units are added.

## Layout

```
src/
├── run.py                 # entry point — arg parsing, dataset/network selection, task loop
├── utils.py               # checkpointing, logging helpers (ACC / BWT)
├── test_lr_and_rho.py     # diagnostic for per-element LRs and rho gradient flow
├── dataloaders/           # one module per dataset
│   ├── mnist2.py, mnist5.py, pmnist.py, cifar.py, mixture.py
├── networks/
│   ├── FC.py              # BayesianLinear — gates --static and --regularization
│   ├── distributions.py   # VariationalPosterior, Prior
│   ├── BayesianConvs.py, BatchNorm.py
│   ├── mlp_grow.py        # used for mnist2 / mnist5 / pmnist
│   └── resnet_grow.py     # used for cifar / mixture
└── train/
    ├── trainer.py         # training loop, growth criterion, ELBO
    └── utils.py           # BayesianSGD optimizer
```

## Setup

```bash
pip install torch numpy tqdm wandb
```

Logs go to W&B under project `continual_growing`. Use `--wandb_mode offline` to disable network sync, or `wandb disabled` in your environment.

## Quick start

```bash
cd src
python run.py --experiment mnist5 --device cuda
```

`--experiment` is required. Choices: `mnist2`, `mnist5`, `pmnist`, `cifar`, `mixture`. Default device is `mps`; pass `--device cuda` or `--device cpu` as appropriate. Dataset files are read from `--data_path` (default `../data/`); checkpoints land in `../checkpoints/<experiment>_<train_mode>/`.

At the end of the run, `utils.print_log_acc_bwt` reports final-task accuracy across all tasks and BWT (backward transfer).

## The four baselines

The same script covers growing/non-growing × Bayesian/deterministic. Toggle with two flags:

|                  | Growing (`--growth_rate 5`) | Non-growing (`--growth_rate 0`) |
|------------------|-----------------------------|---------------------------------|
| **Bayesian**     | *default*                   | `--growth_rate 0`               |
| **Deterministic**| `--static --growth_rate 5`  | `--static --growth_rate 0`      |

- **`--static`** ([networks/FC.py:158](src/networks/FC.py#L158)) — skips posterior sampling in `BayesianLinear.forward`; the network uses only the means. Also changes optimizer setup ([train/trainer.py:146](src/train/trainer.py#L146)).
- **`--growth_rate N`** ([train/trainer.py:228](src/train/trainer.py#L228)) — number of hidden units added per growth step. `0` disables growth entirely.

Note: the growth criterion is rho-saturation-based ([train/trainer.py:240-267](src/train/trainer.py#L240-L267)), so growth is still driven by Bayesian uncertainty even in `--static` mode (rho is trained but ignored at inference). For a clean deterministic-grow baseline, be aware of this coupling.

When running `--static`, set `--samples 1` to skip the redundant MC loop in the eval ELBO.

## Common knobs

```
--cl_mode {task-incremental, domain-incremental}
--epochs              (default 10)
--sbatch              (default 64)
--hidden_n            (default 16; was 1200 in original setup)
--layers              {1, 2}
--lr_mu, --lr_sigma   (default 0.01)
--samples             MC samples for ELBO (default 5)
```

Bayesian-only:
```
--regularization {bbb, sns, unimodal}     # KL/prior style; see networks/FC.py:48
--sigma_init          posterior init stdev (default 0.1)
--sigma_prior1, --sigma_prior2, --pi      scale-mixture-Gaussian prior
--rho_init_mode {gaussian, bimodal}
```

Growth-only:
```
--growth_rate         neurons added per step (0 disables)
--growth_saturation   fraction of new params that must saturate to trigger growth (default 0.2)
--growth_threshold    stdev below which a param is "saturated" (default 0.05)
```

## Resuming

```bash
python run.py --experiment mnist5 --resume yes --sti 3
```

Loads `model_3.pth.tar` from the checkpoint dir and continues from task 3.

## Diagnostics

```bash
python test_lr_and_rho.py
```

Verifies (1) `update_lr` produces genuinely per-element learning rates, (2) `BayesianSGD` applies them, (3) `weight_rho` receives gradients.

## Notes

- `--train_mode joint` in [run.py:144](src/run.py#L144) concatenates data across tasks but no separate trainer module is wired up; treat this code path as experimental and use the default `grow` mode.
