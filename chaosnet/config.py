# chaosnet/config.py

from dataclasses import dataclass

@dataclass
class ChaosNeuronParams:
    threshold: float = 0.5      # spike threshold
    noise_std: float = 0.05     # gaussian noise
    fail_prob: float = 0.1      # probability spike fails to transmit
    decay: float = 0.02         # membrane decay per step
    refractory_decay: float = 0.9  # how fast refractory effect fades

@dataclass
class ChaosLayerParams:
    input_size: int
    hidden_size: int
    neuron: ChaosNeuronParams

@dataclass
class CortexParams:
    input_size: int
    hidden_sizes: list
    neuron: ChaosNeuronParams
