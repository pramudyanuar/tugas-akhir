"""Shared optimizer utilities for multiprocessing training."""

import torch


class SharedAdam(torch.optim.Adam):
    """Adam optimizer with shared state for Hogwild-style training."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'].share_memory_()
