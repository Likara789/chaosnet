"""
chaosnet/training/losses.py
---------------------------------
Why this exists
- Central place for supervised loss and simple spike-activity regularizers.

How it works
- classification_loss wraps cross-entropy.
- firing_rate_loss computes the absolute deviation from a target firing rate
  across layers to encourage stable, sparse spiking.
"""

# chaosnet/training/losses.py

import torch
from torch import nn
from torch.nn import functional as F


def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Standard cross-entropy classification loss.
    logits: (batch, num_classes)
    targets: (batch,) class indices
    """
    return F.cross_entropy(logits, targets)


def firing_rate_loss(
    spikes_per_layer: list[torch.Tensor],
    target_rate: float = 0.1,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Regularizer that pushes average firing rate toward target_rate.
    spikes_per_layer: list of (batch, hidden_size) spike tensors
    """
    if not spikes_per_layer:
        return torch.tensor(0.0, device="cpu")

    losses = []
    for s in spikes_per_layer:
        rate = s.mean()
        losses.append((rate - target_rate).abs())

    stacked = torch.stack(losses)
    return weight * stacked.mean()
