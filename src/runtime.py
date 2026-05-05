"""Shared training entry: builds data, network, trainer; runs the task loop.

Both `src/run.py` (plain run) and `src/hpo/runner.py` (per-trial) call
`run_training(args, ...)`. The caller owns wandb lifecycle (init/finish).

`on_task_done(t, acc_so_far)` is invoked after each task completes so the
HPO runner can call `trial.report(...) / trial.should_prune()`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

import utils


def _seed_everything(args: Any) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif 'mps' in args.device and torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)


def _select_dataloader(args: Any):
    if args.experiment == 'mnist2':
        from dataloaders import mnist2 as dl
    elif args.experiment == 'mnist5':
        from dataloaders import mnist5 as dl
    elif args.experiment == 'pmnist':
        from dataloaders import pmnist as dl
    elif args.experiment == 'cifar':
        from dataloaders import cifar as dl
    elif args.experiment == 'mixture':
        from dataloaders import mixture as dl
    else:
        raise ValueError(f"unknown experiment: {args.experiment!r}")
    return dl


def _select_network(args: Any):
    if args.experiment in ('mnist2', 'pmnist', 'mnist5'):
        from networks import mlp_grow as network
    else:
        from networks import resnet_grow as network
    return network


def _select_trainer_module(args: Any):
    if args.train_mode == 'grow':
        from train import trainer as trainer_module
        return trainer_module
    raise ValueError(f"unsupported train_mode: {args.train_mode!r}")


def run_training(
    args: Any,
    *,
    on_task_done: Callable[[int, float], None] | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run the full continual-learning task loop.

    Returns `(avg_acc, bwt, acc_matrix, lss_matrix)` matching what
    `utils.print_log_acc_bwt` computes/saves.

    `on_task_done(t, running_avg)`: optional callback invoked after each
    task completes with `running_avg = mean(acc[t, 0..t])`. Used by the
    HPO runner for per-task pruning.
    """
    args.rho = math.log(math.expm1(args.sigma_init))
    args.sig1 = -math.log(args.sigma_prior1)
    args.sig2 = -math.log(args.sigma_prior2)
    utils.print_arguments(args)

    _seed_everything(args)
    print('Using device:', args.device)
    args.checkpoint = utils.make_directories(args)

    dataloader = _select_dataloader(args)
    trainer_module = _select_trainer_module(args)
    network = _select_network(args)

    print('Load data...')
    data, taskcla, inputsize = dataloader.get(
        data_path=args.data_path, seed=args.seed, valid_split=args.valid_split,
    )
    if args.valid_split > 0:
        print('Per-task valid sizes:')
        for t, _ in taskcla:
            print('  task {}: train={}, valid={}'.format(
                t, data[t]['train']['x'].size(0), data[t]['valid']['x'].size(0)
            ))
    print('Input size =', inputsize, '\nTask info =', taskcla)
    args.num_tasks = len(taskcla)
    args.inputsize, args.taskcla = inputsize, taskcla

    print('Inits...')
    model = network.Net(args).to(args.device)
    try:
        import wandb
        wandb.watch(model)
    except Exception:
        # wandb may be uninitialised in the HPO runner if wandb_mode=disabled.
        pass

    print('-' * 100)
    trainer = trainer_module.Trainer(model, args=args)
    print('-' * 100)

    if args.resume == 'yes':
        ckpt = torch.load(os.path.join(args.checkpoint, f'model_{args.sti}.pth.tar'))
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device=args.device)
    else:
        args.sti = 0

    n_tasks = len(taskcla)
    acc = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    lss = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    xtrain = ytrain = xvalid = yvalid = task_t = task_v = None
    for t, ncla in taskcla[args.sti:]:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        if args.train_mode == 'joint':
            if t == 0:
                xtrain = data[t]['train']['x']
                ytrain = data[t]['train']['y']
                xvalid = data[t]['valid']['x']
                yvalid = data[t]['valid']['y']
                task_t = t * torch.ones(xtrain.size(0)).int()
                task_v = t * torch.ones(xvalid.size(0)).int()
                task = [task_t, task_v]
            else:
                xtrain = torch.cat((xtrain, data[t]['train']['x']))
                ytrain = torch.cat((ytrain, data[t]['train']['y']))
                xvalid = torch.cat((xvalid, data[t]['valid']['x']))
                yvalid = torch.cat((yvalid, data[t]['valid']['y']))
                task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
                task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
                task = [task_t, task_v]
        else:
            xtrain = data[t]['train']['x'].to(args.device)
            ytrain = data[t]['train']['y'].to(args.device)
            xvalid = data[t]['valid']['x'].to(args.device)
            yvalid = data[t]['valid']['y'].to(args.device)
            task = t

        trainer.train(task, xtrain, ytrain, xvalid, yvalid)
        print('-' * 100)

        for u in range(t + 1):
            xtest = data[u]['test']['x'].to(args.device)
            ytest = data[u]['test']['y'].to(args.device)
            test_loss, test_acc = trainer.eval(u, xtest, ytest, debug=True)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(
                u, data[u]['name'], test_loss, 100 * test_acc,
            ))
            acc[t, u] = test_acc
            lss[t, u] = test_loss

        print('Save at ' + args.checkpoint)
        np.savetxt(
            os.path.join(args.checkpoint, f'{args.experiment}_{args.train_mode}_{args.seed}.txt'),
            acc, '%.5f',
        )

        if on_task_done is not None:
            running_avg = float(np.mean(acc[t, : t + 1]))
            on_task_done(t, running_avg)

    avg_acc, bwt = utils.print_log_acc_bwt(args, acc, lss)
    return float(avg_acc), float(bwt), acc, lss
