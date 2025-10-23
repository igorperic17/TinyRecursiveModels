"""
macOS-compatible replacement for adam-atan2 optimizer.
This is a simplified implementation that works without CUDA.
"""

import torch
import math
from torch.optim import Optimizer


class AdamATan2(Optimizer):
    """
    Adam optimizer with atan2-based learning rate scheduling.
    This is a macOS-compatible version that doesn't require CUDA.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamATan2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamATan2, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Atan2-based learning rate adjustment
                # This is a simplified version of the atan2-based scheduling
                step_size = group['lr']
                
                # Simple atan2-inspired adjustment
                if state['step'] > 1:
                    # Use the ratio of current and previous gradients for atan2-like behavior
                    grad_norm = grad.norm()
                    if grad_norm > 0:
                        # Simple atan2-inspired adjustment
                        atan2_factor = math.atan2(exp_avg.norm().item(), grad_norm.item())
                        step_size = step_size * (1.0 + 0.1 * atan2_factor)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# For compatibility, also provide the original class name
AdamATan2Optimizer = AdamATan2
