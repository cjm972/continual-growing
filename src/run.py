import sys,os,argparse,time
import numpy as np
import torch
import utils
from datetime import datetime
import wandb

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',               default=0,              type=int,   help='(default=%(default)d)')
parser.add_argument('--device',             default='mps',       type=str,   help='gpu id')
parser.add_argument('--wandb_mode',         default='online',       type=str,   help='Online or offline for wandb')
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

args=parser.parse_args()

import math
args.rho = math.log(math.expm1(args.sigma_init))
args.sig1 = -math.log(args.sigma_prior1)
args.sig2 = -math.log(args.sigma_prior2)
utils.print_arguments(args)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if 'cuda' in args.device and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif 'mps' in args.device and torch.backends.mps.is_available():
    torch.mps.manual_seed(args.seed)


print('Using device:', args.device)
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint
print()

# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='mnist5':
    from dataloaders import mnist5 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader

# Args -- Trainer
if args.train_mode=='grow':
    from train import trainer as trainer_module

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment == 'mnist5':
    from networks import mlp_grow as network
else:
    from networks import resnet_grow as network


########################################################################################################################
print()

# Initialize weights and biases
wandb.init(project="continual_growing", config=args, mode=args.wandb_mode)

print("Starting this run on :")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

# Load
print('Load data...')
data,taskcla,inputsize=dataloader.get(data_path=args.data_path, seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)
args.num_tasks=len(taskcla)
args.inputsize, args.taskcla = inputsize, taskcla

# Inits
print('Inits...')
model=network.Net(args).to(args.device)
wandb.watch(model)

print('-'*100)
trainer=trainer_module.Trainer(model,args=args)
print('-'*100)

# args.output=os.path.join(args.results_path, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
print('-'*100)

if args.resume == 'yes':
    checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=args.device)
else:
    args.sti = 0


# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla[args.sti:]:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.train_mode == 'joint':
        # Get data. We do not put it to GPU
        if t==0:
            xtrain=data[t]['train']['x']
            ytrain=data[t]['train']['y']
            xvalid=data[t]['valid']['x']
            yvalid=data[t]['valid']['y']
            task_t=t*torch.ones(xtrain.size(0)).int()
            task_v=t*torch.ones(xvalid.size(0)).int()
            task=[task_t,task_v]
        else:
            xtrain=torch.cat((xtrain,data[t]['train']['x']))
            ytrain=torch.cat((ytrain,data[t]['train']['y']))
            xvalid=torch.cat((xvalid,data[t]['valid']['x']))
            yvalid=torch.cat((yvalid,data[t]['valid']['y']))
            task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
            task=[task_t,task_v]
    else:
        # Get data
        xtrain=data[t]['train']['x'].to(args.device)
        ytrain=data[t]['train']['y'].to(args.device)
        xvalid=data[t]['valid']['x'].to(args.device)
        yvalid=data[t]['valid']['y'].to(args.device)
        task=t

    # Train
    trainer.train(task,xtrain,ytrain,xvalid,yvalid)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].to(args.device)
        ytest=data[u]['test']['y'].to(args.device)
        test_loss,test_acc=trainer.eval(u,xtest,ytest,debug=True)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at '+args.checkpoint)
    np.savetxt(os.path.join(args.checkpoint,'{}_{}_{}.txt'.format(args.experiment,args.train_mode,args.seed)),acc,'%.5f')


utils.print_log_acc_bwt(args, acc, lss)
print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

