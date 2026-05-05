import torch
import numpy as np
from .FC import BayesianLinear


class BayesianMLP(torch.nn.Module):
    
    def __init__(self, args):
        super(BayesianMLP, self).__init__()

        ncha,size,_= args.inputsize
        self.taskcla= args.taskcla
        self.samples = args.samples
        self.device = args.device
        self.sbatch = args.sbatch
        # self.init_lr = args.lr
        # dim=60  #100k
        # dim=1200
        dim=args.hidden_n
        layers=args.layers

        self.fc1 = BayesianLinear(ncha*size*size, dim, args)
        if layers==2:
            self.fc2 = BayesianLinear(dim, dim, args)

        self.args = args
        self.cl_mode = getattr(args, 'cl_mode', 'task-incremental')
        self.classifier = torch.nn.ModuleList()
        if self.cl_mode == 'domain-incremental':
            out_dim = max([n for _, n in self.taskcla])
            self.classifier.append(BayesianLinear(dim, out_dim, args))
        else:
            for t,n in self.taskcla:
                self.classifier.append(BayesianLinear(dim, n, args))


    def prune(self,mask_modules):
        for module, mask in mask_modules.items():
            module.prune_module(mask)


    def forward(self, x, sample=False):
        x = x.view(x.size(0),-1)
        z = self.fc1(x, sample)
        
        is_static = getattr(self.args, 'static', False)
        if getattr(self.args, 'successive_inhibition', False) and not is_static:
            # 1. Estimate empirical variance
            num_samples = getattr(self.args, 'inhibition_samples', 5)
            z_samples = torch.stack([self.fc1(x, sample=True) for _ in range(num_samples)])
            z_var = z_samples.var(dim=0, unbiased=False)
            
            # 2. Calculate confidence
            c = 1.0 / (z_var + 1e-6)
            
            # 3. Calculate inhibition term
            P = c * torch.nn.functional.relu(z)
            
            # 4. Apply directional inhibition (from older to newer neurons)
            S = torch.cumsum(P, dim=-1) - P
            
            # Asymmetric Constraint: Old neurons should not inhibit each other to preserve function.
            # Only 'New' neurons (added in current task) are forced to yield to older ones.
            if hasattr(self.fc1, 'weight_mask_new'):
                # weight_mask_new is (out_features, in_features); any(dim=-1) identifies new rows/neurons
                is_new = self.fc1.weight_mask_new.any(dim=-1)
                S = S * is_new.float()
            
            # 5. Final activation
            gamma = getattr(self.args, 'gamma_inhibition', 1.0)
            x = torch.nn.functional.relu(z - gamma * S)
        else:
            x = torch.nn.functional.relu(z)
            
        y=[]
        if getattr(self, 'cl_mode', 'task-incremental') == 'domain-incremental':
            out = self.classifier[0](x, sample)
            for t,i in self.taskcla:
                y.append(out)
        else:
            for t,i in self.taskcla:
                y.append(self.classifier[t](x, sample))
        return [torch.nn.functional.log_softmax(yy, dim=1) for yy in y]


def Net(args):
    return BayesianMLP(args)

