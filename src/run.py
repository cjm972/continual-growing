import argparse
import time
from datetime import datetime

import wandb

from runtime import run_training

tstart = time.time()

# Arguments
parser = argparse.ArgumentParser(description='xxx', epilog='Concrete defaults live in configs/defaults.yaml when --config is used.')
parser.add_argument('--seed',               default=0,              type=int,   help='(default=%(default)d)')
parser.add_argument('--device',             default='mps',       type=str,   help='gpu id')
parser.add_argument('--wandb_mode',         default='online',       type=str,   choices=['online', 'offline', 'disabled'], help='Online/offline/disabled for wandb')
parser.add_argument('--experiment',         default='mnist5',       type=str,   required=True,
                                            choices=['mnist2','mnist5','pmnist','cifar','mixture'])
parser.add_argument('--train_mode',         default='grow',            type=str,   help='continual growth or static joint')
parser.add_argument('--data_path',          default='../data/',            type=str,   help='gpu id')
parser.add_argument('--cl_mode',            default='domain-incremental', type=str, choices=['task-incremental', 'domain-incremental'], help='Continual learning mode')

# Training parameters
parser.add_argument('--output',             default='',                     type=str,   help='')
parser.add_argument('--checkpoint_dir',     default='../checkpoints/',    type=str,   help='')
parser.add_argument('--epochs',            default=10,            type=int,   help='')
parser.add_argument('--sbatch',             default=64,             type=int,   help='')
parser.add_argument('--lr_mu',              default=0.01,           type=float, help='Learning rate for means (mu)')
parser.add_argument('--lr_sigma',           default=0.01,           type=float, help='Learning rate for uncertainties (rho)')
parser.add_argument('--layers',            default=1,              type=int,   help='')
parser.add_argument('--hidden_n',            default=16,           type=int, help='') # default: 1200
parser.add_argument('--parameter',          default='',             type=str,   help='')

# Bayesian hyper-parameters
parser.add_argument('--samples',            default=10,           type=int,     help='Number of Monte Carlo samples')
parser.add_argument('--sigma_init',         default=0.1,       type=float,   help='Initial standard deviation (posterior)')
parser.add_argument('--sigma_prior1',       default=1.0,            type=float,   help='Stdev for the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sigma_prior2',       default=0.00001,       type=float,   help='Stdev for the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi',                 default=0.25,         type=float,   help='weighting factor for prior')
parser.add_argument('--rho_init_mode',      default='gaussian',   type=str,     choices=['gaussian', 'bimodal'], help='Initialization mode for uncertainties (rho)')
parser.add_argument('--regularization',     default='sns',        type=str,     choices=['bbb', 'sns', 'unimodal'], help='Regularization method: bbb (Bayes by Backprop), sns (Spike & Slab on stdev), unimodal (Gaussian prior)')
parser.add_argument('--arch',               default='mlp',          type=str,     help='Bayesian Neural Network architecture')
parser.add_argument('--static',            action='store_true',    help='Use only means (no Bayesian sampling)')

# Growth hyper-parameters
parser.add_argument('--growth_rate',        default=5,              type=int,     help='Number of neurons to add per growth step (0 to disable)')
parser.add_argument('--growth_saturation',  default=0.2,            type=float,   help='Fraction of new parameters that must be saturated to grow')
parser.add_argument('--growth_threshold',   default=0.05,           type=float,   help='Stdev threshold below which a parameter is considered saturated')

parser.add_argument('--resume',          default='no',            type=str,   help='resume?')
parser.add_argument('--sti',             default=0,               type=int,   help='starting task?')
parser.add_argument('--valid_split',     default=0.0,             type=float, help='Fraction of train to hold out as validation (per task). 0.0 keeps legacy valid==train behavior.')
parser.add_argument('--config',          default=[],              nargs='+',  help='YAML config file(s) under configs/. Layered left-to-right (last wins). CLI flags win over configs.')

args = parser.parse_args()

# Apply layered --config overrides (plain-run mode; !hpo: tags would raise here).
if args.config:
    from hpo.config import load_and_apply_configs
    load_and_apply_configs(args, parser, args.config)

wandb.init(project="continual_growing", config=args, mode=args.wandb_mode)
print("Starting this run on :")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

run_training(args)
wandb.finish()

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
