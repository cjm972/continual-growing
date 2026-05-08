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
        self.n_layers = args.layers
        # self.init_lr = args.lr
        # dim=60  #100k
        # dim=1200
        self.cl_mode = getattr(args, 'cl_mode', 'task-incremental')
        self.core_size = args.hidden_n
        self.growth_rate = getattr(args, 'growth_rate', 5)
        
        dim=args.hidden_n
        layers=args.layers

        self.fc1 = BayesianLinear(ncha*size*size, dim, args)
        self.fc_hidden = []
        for i in range(self.n_layers):
            self.fc.append(BayesianLinear(dim, dim, args))

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
        for i in range(self.n_layers):
            z = self.fc_hidden[i](z, sample)
        
        # Successive inhibition (experimental feature)
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
            
            # 4. Apply Block-wise directional inhibition
            H = self.core_size
            G = self.growth_rate
            N = P.shape[-1]
            
            S = torch.zeros_like(P)
            if N > H:
                # Sum of pressures in the core (Block 0)
                B0 = P[:, :H].sum(dim=-1, keepdim=True) # (batch, 1)
                
                # Sum of pressures in subsequent blocks (1, 2, ...)
                P_rem = P[:, H:]
                num_rem = P_rem.shape[-1]
                num_blocks_rem = num_rem // G
                
                if num_blocks_rem > 0:
                    # Group into blocks of size G and sum their pressures
                    P_blocks = P_rem[:, :num_blocks_rem * G].view(P.shape[0], num_blocks_rem, G)
                    B_rem = P_blocks.sum(dim=-1) # (batch, num_blocks_rem)
                    
                    B_all = torch.cat([B0, B_rem], dim=-1) # (batch, 1 + num_blocks_rem)
                    B_cum = torch.cumsum(B_all, dim=-1)
                    
                    # S_blocks[m] is the total inhibition applied to Block m.
                    # S_blocks = [0, B0, B0+B1, ...]
                    S_blocks = torch.cat([torch.zeros_like(B0), B_cum[:, :-1]], dim=-1)
                    
                    # Expand back to individual neurons
                    # Block 0 (core) always has 0 inhibition
                    S[:, :H] = 0
                    
                    # Blocks 1, 2, ... each neuron in block k gets S_blocks[k]
                    S_rem = S_blocks[:, 1:].repeat_interleave(G, dim=-1)
                    S[:, H : H + S_rem.shape[-1]] = S_rem[:, :num_rem]

            # 5. Final activation
            gamma = getattr(self.args, 'gamma_inhibition', 1.0)
            x = torch.nn.functional.relu(z - gamma * S)
        else:
            x = torch.nn.functional.relu(z)
        
        # Soft Winner-Take-All: keep only top 30% of activated neurons
        if getattr(self.args, 'soft_wta', False) and not is_static:
            k = max(1, int(0.3 * x.shape[-1]))
            # Find the k-th largest activation value per sample
            topk_vals, _ = torch.topk(x, k, dim=-1)
            threshold = topk_vals[:, -1:]  # (batch, 1) — the k-th value
            # Zero out neurons below the threshold (mask is differentiable w.r.t. x)
            x = x * (x >= threshold).float()

            
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

