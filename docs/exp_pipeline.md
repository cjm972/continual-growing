# Experimental Pipeline: Growing vs Non-Growing Networks (Non-Bayesian)

Plan for the first round of experiments comparing growing and non-growing
networks under continual learning. Both arms are deterministic (`--static`).
Downstream analysis target: representation geometry.

## Hypothesis

Under task switching, a network that grows its hidden width when training loss
plateaus retains more of its earlier-task representational structure than a
fixed-width network of the same initial size, even without explicit
continual-learning regularization.

## Arms

| Arm        | Flags                                          |
|------------|------------------------------------------------|
| Grow       | `--static --growth_rate 5 --growth_trigger loss_plateau` |
| No-grow    | `--static --growth_rate 0`                     |

Both arms run under naive SGD — no EWC, no replay — for the first pass. This
deliberately stacks the comparison toward "growth helps because new neurons
aren't being overwritten." That's a known weakness of round 1; round 2 adds a
replay buffer to the no-grow arm so the fixed network has something real to
defend.

## Growth trigger: loss-plateau

Replaces the rho-saturation trigger ([trainer.py:240-267](../src/train/trainer.py#L240-L267))
when `--growth_trigger loss_plateau` is set. The rho-saturation path stays
intact for the Bayesian baselines.

Mirrors the existing LR-decay patience block at
[trainer.py:109-123](../src/train/trainer.py#L109-L123).

Rules:
- Track validation loss (already computed each epoch).
- **Patience**: 2 epochs without improvement triggers growth.
- **Cooldown**: 1 epoch after a growth event before the trigger can re-fire.
- **Best-loss reset**: after growth, reset `best_loss` to current valid loss —
  otherwise freshly-added neurons can never beat the pre-growth bound.
- **Max growth events per task**: 3 (cap to avoid runaway).
- **Max width**: `--max_width 256` cap on hidden layer.

New CLI flags:
```
--growth_trigger {rho_saturation, loss_plateau}   # default: rho_saturation
--growth_patience 2
--growth_cooldown 1
--max_growth_events 3
--max_width 256
```

## Capacity fairness

Explicitly de-scoped for round 1. The grown network will end with more
parameters than the fixed one, so headline ACC/BWT will be capacity-confounded.

This is acceptable because the downstream analysis is representation geometry
(CKA, effective rank, neuron overlap) — those metrics are dimensionality-aware
and the geometric story does not hinge on matched capacity.

A `--match_capacity {none, final, initial}` flag will be added as a stub so
the option is reachable later, but only `none` will be implemented in round 1.
Document this caveat in the results section.

## Network defaults

Accept current defaults from [run.py](../src/run.py):
- `--hidden_n 16` (initial width)
- `--layers 1`
- `--growth_rate 5` (neurons added per event)
- Zero-mean init for new neurons (already in `mlp_grow.py`)
- `--epochs 10` per task
- `--sbatch 64`

## Dataset

`mnist5` only for round 1. Two CL modes:
- `--cl_mode task-incremental`
- `--cl_mode domain-incremental`

## Metrics

Accept the existing `print_log_acc_bwt` output from [utils.py](../src/utils.py):
final ACC across all tasks, BWT (backward transfer). Pipe to W&B as today
and copy the final tables into the results section verbatim.

Representation-geometry hooks (saving hidden activations at task boundaries)
are out of scope for round 1. Add in round 2.

## Run grid

| Axis          | Values                              |
|---------------|-------------------------------------|
| Growth        | grow, no-grow                       |
| CL mode       | task-incremental, domain-incremental|
| Seed          | 0, 1, 2, 3, 4                       |
| Dataset       | mnist5                              |

20 runs total. Report mean ± std per cell.

## Implementation order

1. Add CLI flags listed above to [run.py](../src/run.py).
2. Implement `loss_plateau` branch in `BayesianTrainer.train` — patience
   counter, cooldown, best-loss reset, max-event cap.
3. Wire `--max_width` cap into the growth path in
   [trainer.py:228](../src/train/trainer.py#L228).
4. Stub `--match_capacity` flag (no-op except logging the intended mode).
5. Run the 20-cell grid.
6. Write results section.

## Round 2 (deferred)

- Add small replay buffer (~30 LOC) for the no-grow arm.
- Add activation-saving hooks at task boundaries for rep-geometry analysis.
- Possibly extend to pmnist or cifar.
- Implement actual capacity-matching modes if a reviewer asks.

## Open questions

- Should the loss-plateau trigger use train loss instead of valid loss when
  the validation set is small (mnist5 splits)? Default to valid; revisit if
  the trigger is noisy.
- Does the cooldown need to span across the LR-decay event? If LR drops and
  loss plateaus immediately after, that's not a real signal to grow.
