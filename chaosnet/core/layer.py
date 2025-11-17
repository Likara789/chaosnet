import torch
from torch import nn
from chaosnet.neuron import spike, ChaosState
from ..config import ChaosNeuronParams


class ChaosLayer(nn.Module):
    """
    Chaos Layer with biological E/I balance:
    - 80% excitatory neurons
    - 20% inhibitory neurons
    Excitatory neurons: positive outgoing weights
    Inhibitory neurons: negative outgoing weights
    """

    def __init__(self, input_size: int, hidden_size: int, neuron_params: ChaosNeuronParams):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.params = neuron_params

        # 20% inhibitory, 80% excitatory
        self.num_inhib = max(1, int(hidden_size * 0.2))
        self.num_excit = hidden_size - self.num_inhib

        # Feedforward weights (input → all neurons)
        self.input_weights = nn.Linear(input_size, hidden_size, bias=False)

        # Recurrent weights (all neurons → all neurons)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize with small values
        nn.init.kaiming_uniform_(self.input_weights.weight, a=0.1)
        nn.init.kaiming_uniform_(self.recurrent.weight, a=0.1)

        # Apply E/I sign structure once at init
        self._apply_ei_signs()

    @torch.no_grad()
    def _apply_ei_signs(self):
        """
        Ensure excitatory rows are >=0 and inhibitory rows are <=0.
        Called at init and during sleep cycles to keep EI structure.
        """
        w = self.recurrent.weight
        # excitatory rows: positive
        w[:self.num_excit, :] = w[:self.num_excit, :].abs()
        # inhibitory rows: negative
        w[self.num_excit:, :] = -w[self.num_excit:, :].abs()

    @torch.no_grad()
    def enforce_ei_constraints(self):
        """
        Public method for sleep/homeostasis:
        re-enforces EI sign constraints without re-randomizing.
        """
        self._apply_ei_signs()

    def init_state(self, batch_size: int, device=None, dtype=torch.float32) -> ChaosState:
        if device is None:
            device = next(self.parameters()).device
        return ChaosState.init(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, state: ChaosState | None = None):
        batch = x.size(0)
        device = x.device
        dtype = x.dtype

        if state is None:
            state = self.init_state(batch, device=device, dtype=dtype)

        mem = state.membrane
        refr = state.refractory

        # Feedforward + noise
        ff = self.input_weights(x)
        if self.params.noise_std > 0:
            noise = torch.randn_like(mem) * self.params.noise_std
        else:
            noise = 0.0

        mem = mem + ff + noise

        # Refractory suppression
        mem = mem * (1.0 - refr)

        # Spikes
        spikes = spike(mem - self.params.threshold)

        # Spike failures
        if self.params.fail_prob > 0:
            mask = (torch.rand_like(spikes) < self.params.fail_prob).float()
            spikes = spikes * (1.0 - mask)

        # Update refractory
        refr = torch.clamp(refr * self.params.refractory_decay + spikes, 0.0, 1.0)

        # Leak
        mem = mem * (1.0 - self.params.decay)

        # Recurrent projection
        syn = self.recurrent(spikes)

        new_state = ChaosState(mem, refr)
        return syn, new_state, spikes
