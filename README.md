# ChaosNet: Exploring Chaotic Dynamics for Multi-Modal Learning

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=PyTorch)](https://pytorch.org/)

ChaosNet is an experimental spiking architecture inspired by chaotic dynamics and stochastic neuron failures. The same chaotic core is reused across text and vision tasks to study multi-modal learning under extreme parameter budgets.

## Quick Summary
- Shared chaotic cortex processes both language and vision; task heads stay small.
- Tested on AG News, MNIST, EMNIST Letters, IMDB, Fashion-MNIST, CIFAR-10, and a symbolic addition stress test.
- Sweet spot often involves high stochastic neuron death (30-50% during train and inference).
- Scope: early-stage research on toy benchmarks; not SOTA and not validated for scale.

---

## Important Disclaimers
- Early research; numbers need independent reproduction.
- Toy benchmarks only; no evidence for ImageNet, modern LLMs, or large-scale tasks.
- Parameter counts refer to the shared chaotic core unless stated otherwise.
- Limited hyperparameter sweeps; results are sensitive to seeds.
- Some runs fail outright (see addition task); cherry-picking risk is real.

---

## Parameter Accounting (what the "K" means)
Parameter counts below cover only the chaotic core (input->hidden feedforward + recurrent weights). Task-specific pieces (token embeddings, CNN stems, readouts) add dataset-dependent overhead.

| Core label | embed_dim | hidden | ticks | fail_prob (default) | ~core params | Scripts |
|------------|-----------|--------|-------|---------------------|--------------|---------|
| Tiny (~1.3K) | 8 | 32 | 15 | 0.5 | ~1.28K | `1K_experiment.py` |
| Small (~5K) | 16 | 64 | 10 | 0.5 | ~5.12K | Set `shared_hidden=64` and `shared_embed_dim=16` in `5K_experiment.py` |
| Standard (~21K) | 32 | 128 | 10 | 0.5 | ~20.5K | Default in `21K_experiment.py` and current `5K_experiment.py` |
| Large (~328K) | 128 | 512 | 5 | 0.3 | ~327.7K | `multimodel_trainin.py` |

**Note:** `5K_experiment.py` currently ships with the 21K core values; lower the dims above if you want the true 5K run.

---

## Experiment Scripts (default settings)
- `multimodel_trainin.py`: AG News, MNIST, EMNIST Letters; shared core 128x512, 5 ticks, `fail_prob=0.3`; 4/4/5 epochs; retention checks before/after training.
- `21K_experiment.py`: AG News, MNIST, EMNIST Letters; shared core 32x128, 10 ticks, `fail_prob=0.5`; 8/8/10 epochs with retention sweeps (3 runs per dataset).
- `5K_experiment.py`: Same defaults as `21K_experiment.py` today; drop to 16x64 if you want the ~5K core.
- `1K_experiment.py`: Six-task curriculum (AG News, IMDB, Fashion-MNIST, MNIST, CIFAR-10, EMNIST Letters); shared core 8x32, 15 ticks, `fail_prob=0.5`; epochs per task = 8/8/6/8/10/10.
- Single-task ablations:
  - `train_language.py`: AG News, embed 128, hidden 256, 5 ticks, `fail_prob=0.999`, 15 epochs.
  - `train_mnist_sleep.py`: MNIST with 5 ticks, hidden 512, `fail_prob=0.999`, 15 epochs, includes sleep/homeostasis steps.
  - `train_addition.py`: Symbolic addition (fails on purpose), embed 64, hidden 512, 10 ticks, `fail_prob` up to 0.999.
  - `examples/train_dummy.py`: XOR sanity check.

---

## Reported Metrics

### ~21K Core (32x128, 10 ticks, fail_prob 0.3-0.5)
| Dataset | Classes | Test Accuracy | Model Size* |
|---------|---------|---------------|-------------|
| AG News | 4 | **89.24%** | ~660K |
| MNIST | 10 | **99.20%** | ~35K |
| EMNIST Letters | 26 | **93.89%** | ~37K |

*Model size includes task-specific heads/embeddings for that dataset.

### ~5K Core Variant (16x64, 10 ticks, fail_prob 0.5)
| Dataset | Classes | Baseline | Final Acc | Improvement | Model Size* |
|---------|---------|----------|-----------|-------------|-------------|
| AG News | 4 | 22.78% | **89.18%** | +66.40 pts | ~642K |
| MNIST | 10 | 8.14% | **99.04%** | +90.90 pts | ~24K |
| EMNIST Letters | 26 | 4.58% | **92.53%** | +87.95 pts | ~26K |

### Tiny Six-Task Suite (~1.3K Core, 8x32, 15 ticks, fail_prob 0.5)
| Dataset | Classes | Best Val | Test Acc | Retention | Improvement |
|---------|---------|----------|----------|-----------|-------------|
| AG News | 4 | 84.04% | **84.08%** | 79.73% | +54.67 pts |
| IMDB | 2 | 32.50% | **30.00%** | 38.75% | +13.33 pts |
| Fashion MNIST | 10 | 83.42% | **83.42%** | 40.31% | +30.31 pts |
| MNIST | 10 | 93.27% | **94.08%** | 65.38% | +55.58 pts |
| CIFAR-10 | 10 | 49.64% | **50.90%** | 42.42% | +32.42 pts |
| EMNIST Letters | 26 | 78.85% | **78.97%** | 78.79% | +74.51 pts |

### Historical Variant (~10K Core, 50-dim embed, 200 hidden, 15 ticks, fail_prob 0.5)
| Dataset | Classes | Best Val | Test Acc | Retention | Improvement |
|---------|---------|----------|----------|-----------|-------------|
| AG News | 4 | 89.49% | **89.34%** | 86.85% | +61.26 pts |
| IMDB | 2 | 100% | **100.00%** | 50.42% | -16.25 pts |
| Fashion MNIST | 10 | 87.47% | **87.47%** | 69.36% | +59.36 pts |
| MNIST | 10 | 97.55% | **97.78%** | 94.14% | +84.41 pts |
| CIFAR-10 | 10 | 79.98% | **81.06%** | 80.14% | +70.14 pts |
| EMNIST Letters | 26 | 89.51% | **89.53%** | 89.49% | +85.67 pts |

---

## Comparison to Standard Architectures (single-task baselines)

| Model Type | AG News | MNIST | EMNIST | Params/Task | Multi-Modal? |
|------------|---------|-------|--------|-------------|--------------|
| ChaosNet (~5K core) | 89.2% | 99.0% | 92.5% | 24-642K | Yes |
| ChaosNet (~21K core) | 89.2% | 99.2% | 93.9% | 35-660K | Yes |
| Simple MLP | ~89% | 98.5% | ~90% | ~50K | No |
| LeNet-5 (1998) | N/A | 99.0% | N/A | 60K | No |
| Small CNN | N/A | 99.2% | ~92% | 50-100K | No |
| TinyBERT | 91%+ | N/A | N/A | 4.4M | No |

**Interpretation:** Competitive on toy datasets, uniquely multi-modal, not superior in absolute accuracy to specialized architectures, and parameter counts depend heavily on modality-specific pieces.

---

## Known Failure Case: Addition (Symbolic Reasoning)

| Metric | Result |
|--------|--------|
| Best accuracy | **26%** |
| Random baseline | ~10% |
| Conclusion | Fails at precise arithmetic despite large cores and many ticks |

ChaosNet performs poorly on algorithmic tasks even with large hidden sizes and high failure rates.

---

## Counter-Intuitive Findings

### Stochastic Failure Sometimes Helps
MNIST @ 5 ticks, 512 hidden:
| Neuron Death Rate | Val Accuracy | Delta vs 0% |
|-------------------|--------------|-------------|
| 0% | 90.08% | baseline |
| 50% | **91.00%** | +0.92 pts |
| 90% | 86.00% | -4.08 pts |
| 99% | 53.00% | -37.08 pts |
| 99.9% | 13.00% | -77.08 pts |

AG News @ 5 ticks, 512 hidden:
| Neuron Death Rate | Val Accuracy | Stability |
|-------------------|--------------|-----------|
| 0% - 97% | ~90% | Wide plateau |
| 99% | 87.12% | Graceful degradation |
| 99.9% | 86.58% | Still functional |

### Abrupt Phase Transitions
Example EMNIST run (epoch 1 -> 2):

| Metric | Epoch 1 | Epoch 2 | Change |
|--------|---------|---------|--------|
| Loss | 2.43 | 0.66 | -73% in one epoch |
| Train Acc | 48% | 82% | +34 pts |
| Val Acc | 74% | 85% | +11 pts |

### Parameter Scaling Paradox
Smaller cores (1-5K) sometimes overfit (train > val), while larger cores (21K+) generalize better - opposite of typical deep nets.

---

## Methodology & Rigor

| Aspect | Status | Details |
|--------|--------|---------|
| Multiple runs | Done | 3 independent seeds per configuration where scripted |
| Standard splits | Done | Official train/val/test for all datasets |
| Code released | Done | Full implementation in this repo |
| Failure cases | Shown | Addition task demonstrates limits |
| Retention tracking | Done | Baseline vs final accuracy logged |
| Hyperparameter tuning | Limited | Coarse grid only |
| Complex benchmarks | Missing | No ImageNet/WikiText scale experiments |
| Architecture comparison | Limited | Few alternative designs explored |

### Reproduction Commands

```bash
# Multi-modal, ~21K core (AG News, MNIST, EMNIST Letters)
python 21K_experiment.py

# Same script with true ~5K core: set shared_hidden=64 and shared_embed_dim=16
python 5K_experiment.py

# Tiny six-task curriculum (~1.3K core)
python 1K_experiment.py

# Larger core, shorter runs (defaults)
python multimodel_trainin.py

# Single-task baselines and failure case
python train_mnist_sleep.py
python train_language.py
python train_addition.py
```

Hardware used: consumer-grade GPUs; scripts default to CUDA when available and CPU otherwise.

---

## Quick Start

```bash
git clone https://github.com/Likara789/chaosnet.git
cd chaosnet
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
# .\venv\Scripts\activate

pip install -r requirements.txt
```

---

## Basic Usage (shared cortex across tasks)

```python
from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex

neuron_params = ChaosNeuronParams(
    threshold=0.5,
    noise_std=0.01,
    fail_prob=0.5,   # stochastic death during train + inference
    decay=0.02,
    refractory_decay=0.95,
)

cortex_params = CortexParams(
    input_size=32,    # match your embedding or projection dim
    hidden_sizes=[128],
    neuron=neuron_params,
)

shared_cortex = ChaosCortex(cortex_params)
# Plug this shared_cortex into language and vision heads (see experiment scripts)
```

---

## Core Concepts

### Temporal Chaotic Dynamics

```python
spikes_accum = torch.zeros(batch, hidden_size)
state = cortex.init_state(batch, device=x.device, dtype=x.dtype)
for _ in range(num_ticks):
    out, state, layer_spikes = cortex(x, state)
    spikes = layer_spikes[0] if layer_spikes else out
    spikes_accum.add_(spikes)
avg_spikes = spikes_accum / num_ticks
logits = readout(avg_spikes)
```

Key properties:
- Stochastic death per forward pass (`fail_prob`) during train and inference.
- Refractory and leak dynamics inside each layer.
- Temporal integration over multiple ticks for robustness.

---

## Project Structure

```
chaosnet/
  core/                # cortex, layers, neuron primitives
  io/                  # dataset helpers
  sim/                 # simulation utilities
  training/            # generic training helpers
  utils/               # misc helpers
examples/              # XOR sanity check
visualize/             # plotting helpers
generate_visualizations.py
plot_training.py
copy_files.ps1
requirements*.txt
setup.py
1K_experiment.py       # six-task tiny core
5K_experiment.py       # mid-size core (currently 21K defaults)
21K_experiment.py      # standard core multi-modal run
multimodel_trainin.py  # larger core baseline
train_*                # single-task ablations and failure case
```

---

## Open Questions
- Will this approach scale to ImageNet, CIFAR-100, or modern NLP benchmarks?
- Why does high failure probability help? Regularization, redundancy, or attractor dynamics?
- Can we predict the abrupt phase transitions seen in some runs?
- Is there a principled way to pick `fail_prob` per task?
- How much information can the chaotic core store without catastrophic forgetting?
- Do sparse spikes translate to real hardware efficiency?

---

## Contributing

We welcome:
- Theoretical analysis of chaotic dynamics and phase transitions
- Scaling studies on harder datasets
- Hardware/edge deployment experiments
- Ablations and hyperparameter searches
- Interpretability/visualization of learned attractors
- Failure analyses and robustness tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for the process.

---

## Citation

If this work is useful for your research:

```bibtex
@misc{chaosnet2025,
  title={ChaosNet: Exploring Chaotic Dynamics for Multi-Modal Learning},
  author={Likara789},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/Likara789/chaosnet}},
  note={Experimental architecture - see limitations in README}
}
```

---

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE)

In brief: you can use, modify, and distribute this code, but running it as a network service requires releasing your source.

---

## FAQ

**Q: Is this better than transformers/CNNs/RNNs?**  
A: No. It is an exploratory architecture on toy problems.

**Q: Will this scale to large models like GPT or ResNet?**  
A: Unknown; no evidence yet.

**Q: Why publish if results are not SOTA?**  
A: Negative and surprising results (e.g., 50% > 0% failure) are still informative.

**Q: Is neuron failure just dropout?**  
A: No. Dropout zeroes activations during training only; ChaosNet samples failure every forward pass, including inference.

**Q: Can I use this in production?**  
A: Not recommended. Expect research-grade stability.

**Q: How should I pick `fail_prob`?**  
A: Start with 0.3-0.5; AG News was robust up to ~97% in some runs. Tune per task.

**Q: Why AGPL instead of MIT/Apache?**  
A: To keep service deployments open-sourced.

---

## Acknowledgments

- PyTorch team for the framework
- ML research community for feedback and inspiration
- Chaos theory and computational neuroscience literature
- Beta testers and early adopters

---

**Last Updated**: November 20, 2025  
**Version**: 1.0.0  
**Experiment Logs**: See `experiments/multimodal/` for training curves, checkpoints, and raw data
