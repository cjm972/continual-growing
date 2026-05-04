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
        self.init_lr = args.lr
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
        x = torch.nn.functional.relu(self.fc1(x, sample))
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

