"""
Diagnostic script for GROW's update_lr and rho dynamics.

Tests three things:
  1. Does update_lr produce genuinely per-element learning rates (i.e. a tensor
     whose entries vary) when rho differs across parameters?
  2. Does BayesianSGD actually apply those per-element LRs in a gradient step?
  3. Does weight_rho actually change during training (i.e. is it in the
     computational graph and receiving gradients)?

Run from the GROW/src directory:
    python test_lr_and_rho.py
"""

import sys, os, argparse, copy
import numpy as np
import torch
import torch.nn.functional as F

# ── Setup: minimal args that mimic run.py ────────────────────────────────────

device = 'cpu'  # deterministic for testing

class FakeArgs:
    device = device
    lr = 0.01
    sbatch = 32
    epochs = 5
    arch = 'mlp'
    samples = 2
    hidden_n = 8          # tiny network
    layers = 1
    rho = -3.0
    sig1 = 0.0
    sig2 = 6.0
    pi = 0.25
    experiment = 'pmnist'
    num_tasks = 2
    inputsize = (1, 4, 4)   # 16-dim input
    taskcla = [(0, 3), (1, 3)]
    output = ''
    checkpoint = '/tmp'

args = FakeArgs()

# ── Build model & trainer ───────────────────────────────────────────────────

from networks.mlp_grow import Net
from train.utils import BayesianSGD

model = Net(args).to(device)
print("="*70)
print("MODEL ARCHITECTURE")
print("="*70)
for n, p in model.named_parameters():
    print(f"  {n:30s}  shape={list(p.shape)}")

# ──────────────────────────────────────────────────────────────────────────────
# TEST 1: Does update_lr produce per-element learning rates?
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 1: Per-element learning rates from update_lr")
print("="*70)

# Manually make rho non-uniform so we can detect per-element LR
with torch.no_grad():
    model.fc1.weight_rho.data = torch.linspace(-5, 0, model.fc1.weight_rho.numel(),
                                                device=device).reshape_as(model.fc1.weight_rho)

# Simulate what trainer.py does for t > 0
from train.trainer import Trainer
trainer = Trainer(model, args)

# t=0 case: should be a single scalar LR
params_dict_t0 = trainer.update_lr(t=0)
print(f"\n--- t=0 (first task) ---")
for i, group in enumerate(params_dict_t0):
    lr = group['lr']
    print(f"  group {i}: lr type={type(lr).__name__}, "
          f"lr={'scalar='+str(lr) if not isinstance(lr, torch.Tensor) else 'tensor shape='+str(list(lr.shape))}")

# t=1 case: should have per-element tensor LRs for weight_mu
params_dict_t1 = trainer.update_lr(t=1)
print(f"\n--- t=1 (second task, uncertainty-scaled LR) ---")
test1_pass = False
for i, group in enumerate(params_dict_t1):
    lr = group['lr']
    param_shape = list(group['params'].shape) if isinstance(group['params'], torch.Tensor) else '(generator)'
    if isinstance(lr, torch.Tensor):
        unique_vals = lr.unique().numel()
        print(f"  group {i}: lr is Tensor shape={list(lr.shape)}, "
              f"unique values={unique_vals}, min={lr.min():.6f}, max={lr.max():.6f}")
        if unique_vals > 1:
            test1_pass = True
    else:
        print(f"  group {i}: lr is scalar = {lr}")

if test1_pass:
    print("\n TEST 1 PASSED: update_lr produces per-element tensor LRs with varying values.")
else:
    print("\n TEST 1 FAILED: All LR values are identical — rho is not producing element-wise rates.")

# ──────────────────────────────────────────────────────────────────────────────
# TEST 2: Does BayesianSGD actually apply per-element LRs?
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 2: BayesianSGD applies per-element learning rates")
print("="*70)

# Create a fresh model for a clean test
model2 = Net(args).to(device)
with torch.no_grad():
    # Set rho non-uniform
    model2.fc1.weight_rho.data = torch.linspace(-5, 0, model2.fc1.weight_rho.numel(),
                                                 device=device).reshape_as(model2.fc1.weight_rho)

trainer2 = Trainer(model2, args)
params_dict = trainer2.update_lr(t=1)
optimizer = BayesianSGD(params=params_dict)

# Snapshot weight_mu before step
w_mu_before = model2.fc1.weight_mu.data.clone()

# Create a synthetic forward/backward pass to generate gradients
x_fake = torch.randn(4, 16, device=device)
y_fake = torch.randint(0, 3, (4,), device=device)
out = model2(x_fake, sample=True)
loss = F.nll_loss(out[0], y_fake)
optimizer.zero_grad()
loss.backward(retain_graph=True)

# Check that weight_mu has a gradient
has_grad = model2.fc1.weight_mu.grad is not None
print(f"\n  weight_mu has gradient: {has_grad}")
if has_grad:
    grad_nonzero = (model2.fc1.weight_mu.grad != 0).sum().item()
    print(f"  Non-zero gradient entries: {grad_nonzero}/{model2.fc1.weight_mu.grad.numel()}")

# Take a step
optimizer.step()

w_mu_after = model2.fc1.weight_mu.data.clone()
delta = (w_mu_after - w_mu_before).abs()

print(f"\n  weight_mu change after one step:")
print(f"    min  Δ = {delta.min():.8f}")
print(f"    max  Δ = {delta.max():.8f}")
print(f"    mean Δ = {delta.mean():.8f}")
print(f"    unique Δ values = {delta.unique().numel()}")

# The key check: if LR is truly per-element, different parameters with the
# same gradient magnitude should get different updates
if delta.unique().numel() > 1 and delta.max() > delta.min() * 1.01:
    print("\n  TEST 2 PASSED: Parameter updates vary per-element, confirming per-element LRs are applied.")
else:
    print("\n  TEST 2 FAILED: Parameter updates are uniform — BayesianSGD may not be applying per-element LRs.")

# Also check: is the tensor LR actually used in the step (line 65-66 of utils.py)?
print(f"\n  Sanity check — BayesianSGD tensor LR branch:")
for i, group in enumerate(optimizer.param_groups):
    lr = group['lr']
    is_tensor = isinstance(lr, torch.Tensor)
    print(f"    group {i}: isinstance(lr, Tensor) = {is_tensor}")
    if is_tensor:
        # Check the tensor LR shape matches the param shape
        for p in [group['params']] if isinstance(group['params'], torch.Tensor) else group['params']:
            if p.shape == lr.shape:
                print(f"      PASS! LR shape {list(lr.shape)} matches param shape {list(p.shape)}")
            else:
                print(f"      WARNING! LR shape {list(lr.shape)} != param shape {list(p.shape)} — broadcast will happen!")

# ──────────────────────────────────────────────────────────────────────────────
# TEST 3: Does weight_rho actually change during training?
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: Does weight_rho change during training?")
print("="*70)

model3 = Net(args).to(device)
trainer3 = Trainer(model3, args)

# Snapshot rho before any training
rho_before = {
    'fc1.weight_rho': model3.fc1.weight_rho.data.clone(),
    'fc1.bias_rho':   model3.fc1.bias_rho.data.clone(),
}
for ci, cls in enumerate(model3.classifier):
    rho_before[f'classifier.{ci}.weight_rho'] = cls.weight_rho.data.clone()

print(f"\n  Initial fc1.weight_rho stats: mean={rho_before['fc1.weight_rho'].mean():.6f}, "
      f"std={rho_before['fc1.weight_rho'].std():.6f}")

# Run a few training steps at t=1 (so update_lr puts rho in the optimizer)
x_train = torch.randn(64, 16, device=device)
y_train = torch.randint(0, 3, (64,), device=device)

n_steps = 20
print(f"\n  Running {n_steps} training steps at t=1...")

for step in range(n_steps):
    params_dict = trainer3.update_lr(t=1)
    optimizer = BayesianSGD(params=params_dict)

    # Mini-batch
    idx = torch.randint(0, 64, (8,))
    x_b, y_b = x_train[idx], y_train[idx]

    out = model3(x_b, sample=True)
    # Compute a full ELBO-like loss so rho is in the graph
    lp = model3.fc1.log_prior + model3.classifier[0].log_prior
    lvp = model3.fc1.log_variational_posterior + model3.classifier[0].log_variational_posterior
    nll = F.nll_loss(out[0], y_b)
    loss = (lvp - lp) * 1e-3 + nll

    optimizer.zero_grad()
    loss.backward(retain_graph=True)

    # Check if rho gets a gradient
    if step == 0:
        rho_grad = model3.fc1.weight_rho.grad
        if rho_grad is not None:
            print(f"  Step 0: weight_rho.grad exists, norm={rho_grad.norm():.8f}")
        else:
            print(f"  Step 0: weight_rho.grad is None! ← rho is NOT in computational graph")

    optimizer.step()

# Compare rho after training
rho_after = {
    'fc1.weight_rho': model3.fc1.weight_rho.data.clone(),
    'fc1.bias_rho':   model3.fc1.bias_rho.data.clone(),
}

print(f"\n  After {n_steps} steps:")
for name in ['fc1.weight_rho', 'fc1.bias_rho']:
    diff = (rho_after[name] - rho_before[name]).abs()
    changed = (diff > 1e-10).sum().item()
    total = diff.numel()
    print(f"    {name}: {changed}/{total} values changed, "
          f"max Δ={diff.max():.8f}, mean Δ={diff.mean():.8f}")

fc1_changed = (rho_after['fc1.weight_rho'] - rho_before['fc1.weight_rho']).abs().max() > 1e-10
if fc1_changed:
    print(f"\n  TEST 3 PASSED: weight_rho changes during training — uncertainty is being learned.")
else:
    print(f"\n  TEST 3 FAILED: weight_rho did NOT change — it's not receiving gradients or not being updated.")

    # Dig deeper: check if it's a requires_grad issue
    print(f"\n  Debugging:")
    print(f"    fc1.weight_rho.requires_grad = {model3.fc1.weight_rho.requires_grad}")
    print(f"    fc1.weight_rho is nn.Parameter = {isinstance(model3.fc1.weight_rho, torch.nn.Parameter)}")

    # Check if rho is even in any param_group
    params_dict = trainer3.update_lr(t=1)
    rho_in_optimizer = False
    for group in params_dict:
        p = group['params']
        if isinstance(p, torch.Tensor) and p is model3.fc1.weight_rho:
            rho_in_optimizer = True
            break
        elif isinstance(p, torch.nn.Parameter) and p is model3.fc1.weight_rho:
            rho_in_optimizer = True
            break
    print(f"    fc1.weight_rho is in optimizer param_groups = {rho_in_optimizer}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 4: Verify the LR tensor shape matches what BayesianSGD expects
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 4: LR tensor shape vs param shape compatibility")
print("="*70)

model4 = Net(args).to(device)
trainer4 = Trainer(model4, args)
params_dict = trainer4.update_lr(t=1)

print()
all_compatible = True
for i, group in enumerate(params_dict):
    lr = group['lr']
    p = group['params']
    p_shape = list(p.shape) if hasattr(p, 'shape') else 'N/A'

    if isinstance(lr, torch.Tensor):
        lr_shape = list(lr.shape)
        compatible = lr.shape == p.shape
        if not compatible:
            all_compatible = False
        status = "YES" if compatible else "MISMATCH"
        print(f"  group {i}: param {p_shape} | lr {lr_shape} | {status}")
    else:
        print(f"  group {i}: param {p_shape} | lr scalar={lr}")

if all_compatible:
    print(f"\n  TEST 4 PASSED: All tensor LR shapes match their parameter shapes.")
else:
    print(f"\n  TEST 4 FAILED: Some LR tensors don't match param shapes — updates may broadcast incorrectly.")


# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
If tests 1 & 2 pass but test 3 fails, then:
  → update_lr correctly computes per-element LRs from rho
  → BayesianSGD correctly applies them
  → BUT rho itself is frozen (not receiving gradients), so the LRs
    never actually change between epochs — they stay at their initial values.

If test 1 fails:
  → The rho values may all be identical (check initialization).

If test 2 fails:
  → BayesianSGD's tensor branch (line 65-66 in utils.py) may have a bug.
""")
