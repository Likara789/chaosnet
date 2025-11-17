# chaosnet/training/utils.py

import torch
from torch import nn
from chaosnet.losses import classification_loss, firing_rate_loss


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    firing_reg_weight: float = 0.0,
    firing_target_rate: float = 0.1,
):
    """
    Generic training loop for one epoch.
    Expects model(x) -> (logits, spikes_per_layer)
    where spikes_per_layer is list[Tensor] or None.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, spikes_per_layer = model(x)

        loss = classification_loss(logits, y)

        if firing_reg_weight > 0.0 and spikes_per_layer is not None:
            fr_loss = firing_rate_loss(
                spikes_per_layer,
                target_rate=firing_target_rate,
                weight=firing_reg_weight,
            )
            loss = loss + fr_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return 0.0

    return total_loss / total_batches
