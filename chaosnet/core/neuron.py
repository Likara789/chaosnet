# chaosnet/core/neuron.py

import torch
from torch import nn

class SpikeFn(torch.autograd.Function):
    """
    Straight-through surrogate gradient spike:
    forward: hard threshold
    backward: smooth triangular window around threshold
    """
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        width = 1.0  # size of region with non-zero gradient

        # mask of inputs near threshold
        mask = (input > threshold - width) & (input < threshold + width)

        grad_input = torch.zeros_like(input)
        if grad_output is not None:
            local_slope = 1.0 - (input - threshold).abs() / width  # triangular
            local_slope = torch.clamp(local_slope, min=0.0)
            grad_input = grad_output * local_slope * mask.float()

        # no gradient for threshold
        return grad_input, None


def spike(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Convenience function: spiking non-linearity with surrogate gradient.
    """
    if not torch.is_tensor(threshold):
        threshold = torch.tensor(threshold, device=x.device, dtype=x.dtype)
    return SpikeFn.apply(x, threshold)


class ChaosState:
    """
    Small container for neuron state inside a layer.
    membrane: current membrane potential
    refractory: refractory level (0..1)
    """
    def __init__(self, membrane: torch.Tensor, refractory: torch.Tensor):
        self.membrane = membrane
        self.refractory = refractory

    def to(self, device):
        self.membrane = self.membrane.to(device)
        self.refractory = self.refractory.to(device)
        return self

    @staticmethod
    def init(batch_size: int, hidden_size: int, device=None, dtype=torch.float32):
        membrane = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        refractory = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        return ChaosState(membrane, refractory)
