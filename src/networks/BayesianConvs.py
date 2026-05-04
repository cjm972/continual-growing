import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .distributions import VariationalPosterior, Prior



class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, use_bias, args):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = use_bias
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi
        self.rho = args.rho
        self.device = args.device
        self.args = args
        self.static = getattr(args, 'static', False)


        if transposed:
            self.weight_mu = nn.Parameter(torch.Tensor(in_channels, out_channels//groups, *kernel_size).normal_(0., 0.1))
            # self.weight_mu = nn.Parameter(torch.normal(mean=0., std=0.1, size=(in_channels, out_channels//groups, *kernel_size)))
            self.weight_rho = nn.Parameter(self.rho + torch.zeros(in_channels, out_channels//groups,*kernel_size).normal_(0., 0.1))

        else:
            self.weight_mu = nn.Parameter(torch.empty((out_channels, in_channels//groups, *kernel_size),
                                     device=self.device, dtype=torch.float32).normal_(0., 0.1), requires_grad=True)
            
            w_shape = (out_channels, in_channels//groups, *kernel_size)
            w_rho_base = self._get_init_rho(w_shape, args)
            self.weight_rho = nn.Parameter(w_rho_base + torch.empty((out_channels, in_channels//groups, *kernel_size),
                                         device=self.device, dtype=torch.float32).normal_(0.,0.1), requires_grad=True)

        self.weight = VariationalPosterior(self.weight_mu, self.weight_rho, self.device).to(self.device)

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((self.out_channels),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
            
            b_rho_base = self._get_init_rho((self.out_channels,), args)
            self.bias_rho = nn.Parameter(b_rho_base + torch.empty((self.out_channels,),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1), requires_grad=True)

            self.bias = VariationalPosterior(self.bias_mu, self.bias_rho, self.device).to(self.device)
        else:
            self.register_parameter('bias', None)            
        
        # Prior distributions
        from .distributions import UnimodalPrior, StdevMixturePrior
        reg_mode = getattr(args, 'regularization', 'bbb')
        if reg_mode == 'unimodal':
            self.weight_prior = UnimodalPrior(args).to(self.device)
            if self.use_bias: self.bias_prior = UnimodalPrior(args).to(self.device)
        elif reg_mode == 'sns':
            self.weight_prior = StdevMixturePrior(args).to(self.device)
            if self.use_bias: self.bias_prior = StdevMixturePrior(args).to(self.device)
        else: # bbb
            self.weight_prior = Prior(args).to(self.device)
            if self.use_bias: self.bias_prior = Prior(args).to(self.device)


    def _get_init_rho(self, shape, args):
        import math
        if getattr(args, 'rho_init_mode', 'gaussian') == 'bimodal':
            rho1 = math.log(math.expm1(args.sigma_prior1))
            rho2 = math.log(math.expm1(args.sigma_prior2))
            mask = (torch.rand(shape, device=self.device) < args.pi).float()
            return mask * rho1 + (1. - mask) * rho2
        else:
            return torch.full(shape, self.rho, device=self.device)

        self.log_prior = torch.tensor(0.0, device=self.device)
        self.log_variational_posterior = torch.tensor(0.0, device=self.device)
        
        self.mask_flag = False


class BayesianConv2D(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, use_bias, args)



    def prune_module(self, mask):
        self.mask_flag = True 
        self.pruned_weight_mu=self.weight_mu.data.mul_(mask)
        # self.pruned_weight_rho=self.weight_rho.data.mul_(mask)
        # pruning_mask = torch.eq(mask, torch.zeros_like(mask))


    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.mask_flag:
            self.weight = VariationalPosterior(self.pruned_weight_mu, self.weight_rho, self.device)
            # if self.use_bias:
            #     self.bias = VariationalPosterior(self.bias_mu, self.bias_rho)

        if (self.training or sample) and not self.static:
            weight = self.weight.sample()
            bias = self.bias.sample() if self.use_bias else None
                
        else:
            weight = self.weight.mu
            bias = self.bias.mu if self.use_bias else None

        if self.training or calculate_log_probs:
            reg_mode = getattr(self.args, 'regularization', 'bbb')
            
            if reg_mode in ['bbb', 'unimodal']:
                if self.use_bias:
                    self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
                    self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
                else:
                    self.log_prior = self.weight_prior.log_prob(weight)
                    self.log_variational_posterior = self.weight.log_prob(weight)
            elif reg_mode == 'sns':
                if self.use_bias:
                    self.log_prior = self.weight_prior.log_prob(self.weight.sigma) + self.bias_prior.log_prob(self.bias.sigma)
                else:
                    self.log_prior = self.weight_prior.log_prob(self.weight.sigma)
                self.log_variational_posterior = torch.tensor(0.0, device=self.device)
        else:
            self.log_prior = torch.tensor(0.0, device=self.device)
            self.log_variational_posterior = torch.tensor(0.0, device=self.device)
        
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
