# 🌌 ChaosNet: Exploring Chaotic Dynamics for Multi-Modal Learning

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=PyTorch)](https://pytorch.org/)

ChaosNet is an experimental neural network architecture inspired by chaotic dynamical systems and biological neuron unreliability. This repository explores whether stochastic neuron failures and temporal dynamics can enable efficient multi-modal learning on simple benchmarks.

## 🎯 Quick Summary

An experimental spiking neural network with chaotic dynamics that can sequentially learn text and image classification with minimal forgetting. Uses a shared 2-4K parameter core across modalities. Achieves ~89% on AG News and ~99% on MNIST with the same computational substrate.

**Key finding**: Counterintuitively robust to neuron failure—50% death rate often outperforms 0% failure rate.

**Scope**: Early research on toy datasets. Not claiming SOTA performance or scalability to complex tasks.

---

## ⚠️ Important Disclaimers

**Read this before evaluating claims:**

- **Early Research**: Initial experiments requiring independent verification
- **Toy Benchmarks Only**: MNIST, AG News, EMNIST—not challenging modern datasets
- **Parameter Counting**: 2-4K refers to shared core only, not total model size
- **No Scaling Evidence**: Untested on ImageNet, large language models, or SOTA benchmarks
- **Biological Inspiration ≠ Practical Advantage**: Neuroscience-inspired but not necessarily superior
- **Cherry-Picking Risk**: Showing successful configurations; some fail completely (see Addition task)
- **Limited Hyperparameter Tuning**: Results may not represent optimal performance

---

## 📊 Benchmark Results

### Understanding Parameter Counts

**Architecture breakdown** (using 2K core as example):

| Component | AG News | MNIST | EMNIST |
|-----------|---------|-------|--------|
| Task-specific input layer | ~640K (embedding) | ~21K (CNN) | ~21K (CNN) |
| **Shared chaotic core** | **2K** | **2K** | **2K** |
| Task-specific readout | ~512 | ~1.3K | ~3.3K |
| **Total per task** | **~642K** | **~24K** | **~26K** |

**Key insight**: The same 2K core processes all modalities. Total memory for 3 tasks: ~692K (vs. ~716K for 3 separate models).

---

### Main Results by Configuration

#### Configuration A: Ultra-Compressed Core (2K params)

**Setup**: 16-dim embeddings, 64 hidden neurons, 10 ticks, 50% failure rate

| Dataset | Classes | Baseline | Final Acc | Improvement | Model Size |
|---------|---------|----------|-----------|-------------|------------|
| AG News | 4 | 22.78% | **89.18%** | +66.40 pts | 642K |
| MNIST | 10 | 8.14% | **99.04%** | +90.90 pts | 24K |
| EMNIST Letters | 26 | 4.58% | **92.53%** | +87.95 pts | 26K |

**Observations**: 
- Near-perfect MNIST accuracy with 24K total parameters
- Competitive AG News performance for simple task
- Strong multi-class letter recognition

---

#### Configuration B: Standard Core (4K params)

**Setup**: 32-dim embeddings, 128 hidden neurons, 5-10 ticks, 30% failure rate

| Dataset | Classes | Test Accuracy | Model Size |
|---------|---------|---------------|------------|
| AG News | 4 | **89.24%** | 660K |
| MNIST | 10 | **99.20%** | 35K |
| EMNIST Letters | 26 | **93.89%** | 37K |

**Observations**:
- Marginal improvement over 2K core
- Still highly parameter-efficient
- Diminishing returns from larger core

---

#### Configuration C: Minimal Core (256 params)

**Setup**: 8-dim embeddings, 16 hidden neurons, 15 ticks, 50% failure rate

| Dataset | Classes | Best Val | Test Acc | Retention | Improvement |
|---------|---------|----------|----------|-----------|-------------|
| AG News | 4 | 84.04% | **84.08%** | 79.73% | +54.67 pts |
| IMDB | 2 | 32.50% | **30.00%** | 38.75% | +13.33 pts |
| Fashion MNIST | 10 | 83.42% | **83.42%** | 40.31% | +30.31 pts |
| MNIST | 10 | 93.27% | **94.08%** | 65.38% | +55.58 pts |
| CIFAR-10 | 10 | 49.64% | **50.90%** | 42.42% | +32.42 pts |
| EMNIST Letters | 26 | 78.85% | **78.97%** | 78.79% | +74.51 pts |

**Training epochs**: AG News (8), IMDB (8), Fashion MNIST (6), MNIST (8), CIFAR-10 (10), EMNIST Letters (10)

---

#### Configuration D: Minimal Core (10K params)

**Setup**: 50-dim embeddings, 200 hidden neurons, 15 ticks, 50% failure rate

| Dataset | Classes | Best Val | Test Acc | Retention | Improvement |
|---------|---------|----------|----------|-----------|-------------|
| AG News | 4 | 89.49 | **89.34%** | 86.85% | +61.26 pts |
| IMDB | 2 | 100% | **100.00%** | 50.42% | -16.25 pts |
| Fashion MNIST | 10 | 87.47% | **87.47%** | 69.36% | +59.36 pts |
| MNIST | 10 | 97.55% | **97.78%** | 94.14% | +84.41 pts |
| CIFAR-10 | 10 | 79.98% | **81.06%** | 80.14% | +70.14 pts |
| EMNIST Letters | 26 | 89.51% | **89.53%** | 89.49% | +85.67 pts |

**Training epochs**: AG News (8), IMDB (8), Fashion MNIST (6), MNIST (8), CIFAR-10 (10), EMNIST Letters (10)

---

**Observations**:
- Extreme compression enables 6-task learning
- Significant catastrophic forgetting on some tasks (Fashion MNIST retention: 40%)
- EMNIST and AG News retain well (>79%)
- IMDB struggles (low absolute performance)

---

### Comparison to Standard Architectures

| Model Type | AG News | MNIST | EMNIST | Params/Task | Multi-Modal? |
|------------|---------|-------|--------|-------------|--------------|
| **ChaosNet (2K core)** | 89.2% | 99.0% | 92.5% | 24-642K | ✅ Yes |
| **ChaosNet (4K core)** | 89.2% | 99.2% | 93.9% | 35-660K | ✅ Yes |
| Simple MLP | ~89% | 98.5% | ~90% | ~50K | ❌ No |
| LeNet-5 (1998) | N/A | 99.0% | N/A | 60K | ❌ No |
| Small CNN | N/A | 99.2% | ~92% | 50-100K | ❌ No |
| TinyBERT | 91%+ | N/A | N/A | 4.4M | ❌ No |

**Interpretation**:
- ✅ **Competitive** on individual tasks vs. single-task specialists
- ✅ **Unique** in multi-modal capability with shared core
- ❌ **Not superior** in raw accuracy on any single task
- ❌ **No evidence** of parameter efficiency advantage over specialized models

---

### Known Failure Cases

#### Addition Task (Symbolic Reasoning)

**Setup**: 10-20 ticks, 512-2048 neurons, all failure rates tested

| Metric | Result |
|--------|--------|
| Best accuracy | **26%** |
| Random baseline | ~10% |
| Conclusion | ❌ **Fails at precise arithmetic** |

**Interpretation**: ChaosNet appears suited for pattern recognition but not algorithmic/symbolic manipulation.

---

## 🔬 Counter-Intuitive Findings

### Finding 1: Stochastic Failure Improves Performance

#### MNIST Results (5 ticks, 512 neurons)

| Neuron Death Rate | Val Accuracy | Δ vs. 0% |
|-------------------|--------------|----------|
| **0% (baseline)** | 90.08% | — |
| **50%** | **91.00%** | **+0.92 pts** ✨ |
| 90% | 86.00% | -4.08 pts |
| 99% | 53.00% | -37.08 pts |
| 99.9% | 13.00% | -77.08 pts |

#### AG News Results (5 ticks, 512 neurons)

| Neuron Death Rate | Val Accuracy | Stability |
|-------------------|--------------|-----------|
| 0% - 97% | ~90% | Remarkably stable plateau |
| 99% | 87.12% | Graceful degradation |
| 99.9% | 86.58% | Still functional |

**Possible Explanations** (speculative):
1. **Implicit regularization**: Similar to dropout but at spike level
2. **Forced redundancy**: Network learns distributed representations
3. **Attractor robustness**: Chaos dynamics create noise-resistant computational states
4. **Overfitting prevention**: Stochasticity prevents memorization of specific patterns

**Analogy**: Like biological neurons with ~40-60% synaptic failure rates—constraints may drive robustness.

---

### Finding 2: Abrupt Phase Transitions

**Unlike smooth gradient descent**, some runs show sudden "crystallization":

**Example: EMNIST Epoch 1→2 transition**

| Metric | Epoch 1 | Epoch 2 | Change |
|--------|---------|---------|--------|
| Loss | 2.43 | 0.66 | **-73% (single epoch!)** |
| Train Acc | 48% | 82% | +34 pts |
| Val Acc | 74% | 85% | +11 pts |

**Interpretation**: System may "lock into" computational attractors rather than gradually optimize. Reminiscent of physical phase transitions.

---

### Finding 3: Parameter Scaling Paradox

| Core Size | Train Acc | Val Acc | Behavior |
|-----------|-----------|---------|----------|
| Small (256-2K) | Higher | Lower | Memorization |
| Large (4K+) | Lower | Higher | Generalization |

**This is backwards** from typical neural networks where:
- More parameters → More overfitting → Train > Val

**Speculation**: Extreme constraints force literal pattern matching; more capacity enables abstract feature learning.

---

## 🧪 Methodology & Reproducibility

### Experimental Rigor

| Aspect | Status | Details |
|--------|--------|---------|
| Multiple runs | ✅ Done | 3 independent seeds per configuration |
| Standard splits | ✅ Done | Official train/val/test for all datasets |
| Code released | ✅ Done | Full implementation in repository |
| Failure cases | ✅ Shown | Addition task demonstrates limits |
| Retention tracking | ✅ Done | Baseline→final accuracy logged |
| Hyperparameter tuning | ⚠️ Limited | Not exhaustively optimized |
| Complex benchmarks | ❌ Missing | Only toy datasets tested |
| Architecture comparison | ⚠️ Limited | Few alternative designs explored |

### Reproduction Commands

```bash
# Main multi-task experiment
python tests.py

# Single-task experiments
python train_mnist_sleep.py --neurons 512 --ticks 5 --fail_prob 0.5
python train_language.py --neurons 512 --ticks 5 --fail_prob 0.3

# Configuration details in tests.py:
# - shared_embed_dim: 16 or 32
# - shared_hidden: 64 or 128
# - shared_ticks: 10
# - fail_prob: 0.3 or 0.5
```

**Hardware**: Consumer-grade GPUs (specific details in `experiments/` directory)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Likara789/chaosnet.git
cd chaosnet
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```python
from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex

# Configure chaos neurons
neuron_params = ChaosNeuronParams(
    threshold=0.5,
    noise_std=0.01,
    fail_prob=0.5,          # 50% stochastic neuron death
    decay=0.02,
    refractory_decay=0.95
)

# Configure cortex (shared across modalities)
cortex_params = CortexParams(
    input_size=32,
    hidden_sizes=[128],
    neuron=neuron_params
)

# Create shared computational substrate
cortex = ChaosCortex(cortex_params)

# Use for multiple tasks with different readout heads
# (See examples in tests.py)
```

---

## 🧠 Core Concepts

### Temporal Chaotic Dynamics

Neurons follow discrete-time chaotic dynamics with stochastic spiking:

```python
# Multi-tick temporal processing
spikes_accum = torch.zeros(batch, hidden_size)
for tick in range(num_ticks):
    out, state, layer_spikes = cortex(input, state)
    spikes = layer_spikes[0] if layer_spikes else out
    spikes_accum.add_(spikes)  # Accumulate over time

# Temporal averaging provides robustness
avg_spikes = spikes_accum / num_ticks
logits = readout(avg_spikes)
```

**Key Properties**:
- **Stochastic death**: Each neuron fails with probability `fail_prob` per forward pass
- **Non-deterministic**: Different neurons fail each time (not fixed dropout mask)
- **Membrane dynamics**: Potential decay and refractory periods
- **Temporal integration**: Multiple ticks allow recurrent processing

### Why Explore This?

**Biological Motivation**:  
Real cortical synapses exhibit 40-60% transmission failure rates. If biology uses unreliable components effectively, perhaps artificial systems can too.

**Hypothesized Advantages** (unproven at scale):
- Implicit regularization through stochasticity
- Hardware fault tolerance for edge deployment
- Energy efficiency (fewer active neurons)
- Shared substrate for multi-modal learning

**Known Disadvantages**:
- Slower convergence than backprop
- Difficult hyperparameter tuning
- Limited theoretical understanding
- Unclear scalability to complex tasks

---

## 📂 Project Structure

```
chaosnet/
├── core/
│   ├── cortex.py          # ChaosCortex: main shared substrate
│   ├── layer.py           # Chaos layer implementations
│   └── neuron.py          # Chaotic neuron dynamics
├── config.py              # Configuration dataclasses
├── tests.py               # Multi-task sequential training (MAIN)
├── train_mnist_sleep.py   # MNIST-specific experiments
├── train_language.py      # AG News experiments
└── multimodel_trainin.py  # Documented full implementation
```

**Entry points**:
- `tests.py` - Run multi-modal experiments
- `train_*.py` - Single-task ablations
- `multimodel_trainin.py` - Reference implementation with detailed docstrings

---

## ❓ Open Questions

### Scientific Questions

1. **Scalability**: Does this approach work on ImageNet, CIFAR-100, or modern NLP benchmarks?
2. **Mechanism**: Why does stochasticity help? Is it regularization, redundancy, or attractor dynamics?
3. **Phase transitions**: Can we predict when sudden learning jumps will occur?
4. **Failure rate**: Is 50% optimal universally, or task-dependent?
5. **Capacity**: What's the theoretical information capacity of chaotic temporal dynamics?
6. **Retention**: With 10+ sequential tasks, does catastrophic forgetting emerge?

### Practical Questions

1. **Hardware efficiency**: Do sparse activations translate to energy savings on actual hardware?
2. **Fault tolerance**: Can this architecture handle permanent neuron damage (not just stochastic)?
3. **Interpretability**: What are the learned chaotic attractors actually computing?
4. **Training stability**: How sensitive are results to initialization and hyperparameters?

---

## 🤝 Contributing

We welcome contributions, especially:

- 🔬 **Theoretical analysis**: Mathematical modeling of phase transitions, attractor dynamics
- 📊 **Scaling studies**: Testing on CIFAR-100, Tiny ImageNet, WikiText, etc.
- ⚡ **Hardware deployment**: Edge device experiments, actual energy measurements
- 🧪 **Ablations**: Which components are essential? Minimal working configurations?
- 📈 **Hyperparameter optimization**: Systematic search for optimal settings
- 🔍 **Interpretability**: Visualizing learned attractors, activation patterns
- 🛠️ **Failure analysis**: Documenting when/why the approach breaks down

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 Citation

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

## 📜 License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE)

**In brief**: You can use, modify, and distribute this code, but if you run it as a network service, you must make your source code available.

---

## ❓ FAQ

**Q: Is this better than transformers/CNNs/RNNs?**  
A: No. This explores alternative architectures on toy problems. Not claiming superiority to established methods.

**Q: Will this scale to large models like GPT or ResNet?**  
A: Unknown. Likely not without major modifications. This is early-stage research.

**Q: Why publish if results aren't state-of-the-art?**  
A: Science benefits from negative/unexpected results. The 50% > 0% failure finding is interesting even if not immediately practical.

**Q: Is neuron failure just dropout?**  
A: No—important differences:
- **Dropout**: Temporarily zeros activations during training only
- **Chaos failure**: Persistent stochastic spike failure during train AND inference
- Different mechanism, different effects, different biological analogy

**Q: What's the "killer app" for this architecture?**  
A: Unknown. Possible niches:
- Edge devices requiring fault tolerance
- Multi-modal learning with severe memory constraints
- Research into biological computation principles

**Q: Can I use this in production?**  
A: Not recommended. This is a research prototype with limited testing. Use at your own risk.

**Q: How do I choose `fail_prob`?**  
A: Start with 0.3-0.5 based on our results. Surprisingly robust across wide range (0-97% for AG News). Task-dependent tuning needed.

**Q: Why AGPL license instead of MIT/Apache?**  
A: To ensure improvements to the architecture remain open-source, especially if deployed as a service.

---

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- ML research community for inspiration and constructive feedback
- Chaos theory and computational neuroscience literature
- Beta testers and early adopters

---

**Last Updated**: November 18, 2025  
**Version**: 1.0.0  
**Experiment Logs**: See `experiments/multimodal/` for complete training curves, checkpoints, and raw data