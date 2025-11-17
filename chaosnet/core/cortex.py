"""
chaosnet/core/cortex.py
---------------------------------
Why this exists
- Encapsulates a stack of ChaosLayer modules to form a small "cortex" that
  can be driven for multiple ticks. The state for each layer is explicit so
  callers can decide whether to maintain temporal continuity (by reusing
  states) or start fresh (by reinitializing states).

How it works
- On construction, we build a list of layers from the provided CortexParams.
- init_state creates per-layer ChaosState tensors (membrane/refractory) sized
  for the batch. This separates parameters (on module) from state (per batch).
- forward accepts an optional list of states. If none is given, it will
  initialize empty states for a single tick use-case. Each layer receives the
  same input for a given tick and returns both its spikes and updated state.
"""

# chaosnet/core/cortex.py

import torch
from torch import nn
from chaosnet.layer import ChaosLayer
from chaosnet.neuron import ChaosState
from ..config import CortexParams, ChaosNeuronParams


class ChaosCortex(nn.Module):
    """
    Stack of chaotic spiking layers.
    Each layer maintains its own state; we pass state in/out explicitly.
    """

    def __init__(self, params: CortexParams):
        super().__init__()
        self.params = params

        layers = []
        in_size = params.input_size
        for h in params.hidden_sizes:
            neuron_params = params.neuron
            layers.append(ChaosLayer(in_size, h, neuron_params))
            in_size = h

        self.layers = nn.ModuleList(layers)

    def init_state(self, batch_size: int, device=None, dtype=torch.float32):
        """
        Returns list[ChaosState], one per layer.
        """
        if device is None:
            device = next(self.parameters()).device

        states: list[ChaosState] = []
        for layer in self.layers:
            states.append(layer.init_state(batch_size, device=device, dtype=dtype))
        return states

    def forward(self, x: torch.Tensor, states: list[ChaosState] | None = None):
        """
        x: (batch, input_size)
        states: list[ChaosState] or None
        returns:
          out: (batch, last_hidden_size)
          new_states: list[ChaosState]
          all_spikes: list[(batch, hidden_size)]
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if states is None:
            states = self.init_state(batch_size, device=device, dtype=dtype)

        new_states: list[ChaosState] = []
        all_spikes: list[torch.Tensor] = []

        h = x
        for layer, state in zip(self.layers, states):
            out, new_state, spikes = layer(h, state)
            new_states.append(new_state)
            all_spikes.append(spikes)
            h = out  # feed to next layer

        return h, new_states, all_spikes
