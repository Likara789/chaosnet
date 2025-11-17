# chaosnet/sim/tick_engine.py

import torch
from ..core.cortex import ChaosCortex
from ..core.neuron import ChaosState


class TickEngine:
    """
    Simple helper to run a ChaosCortex over time while keeping internal state.
    """

    def __init__(self, cortex: ChaosCortex, device: str | None = None):
        self.cortex = cortex
        if device is not None:
            self.cortex.to(device)
        self.device = device or next(cortex.parameters()).device
        self.states: list[ChaosState] | None = None

    def reset(self, batch_size: int, dtype=torch.float32):
        self.states = self.cortex.init_state(batch_size, device=self.device, dtype=dtype)

    def step(self, x: torch.Tensor):
        """
        One temporal step:
        x: (batch, input_size)
        returns:
          out, spikes_per_layer
        """
        x = x.to(self.device)
        if self.states is None:
            self.reset(x.size(0), dtype=x.dtype)

        out, new_states, spikes = self.cortex(x, self.states)
        self.states = new_states
        return out, spikes
