import time
from datetime import datetime

import wandb

from params import build_parser
from runtime import run_training

tstart = time.time()

parser = build_parser(description="Continual learning with growing Bayesian nets.")
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
