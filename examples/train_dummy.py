# examples/train_dummy.py

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import os
import sys

# Make sure we can import chaosnet when running this file directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex
from chaosnet.training.utils import train_epoch


class ChaosXORModel(nn.Module):
    def __init__(self):
        super().__init__()

        neuron_params = ChaosNeuronParams(
            threshold=0.3,
            noise_std=0.05,
            fail_prob=0.1,
            decay=0.05,
            refractory_decay=0.9,
        )

        cortex_params = CortexParams(
            input_size=2,
            hidden_sizes=[16, 16],
            neuron=neuron_params,
        )

        self.cortex = ChaosCortex(cortex_params)
        self.readout = nn.Linear(16, 2)  # from last layer hidden to 2 classes (0, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 2)
        returns:
          logits: (batch, 2)
          spikes_per_layer: list[Tensor]
        """
        # We treat one forward pass as one "tick" of the cortex
        out, states, spikes_per_layer = self.cortex(x)
        logits = self.readout(out)
        return logits, spikes_per_layer


def build_xor_dataset():
    """
    XOR:
      0 xor 0 = 0
      0 xor 1 = 1
      1 xor 0 = 1
      1 xor 1 = 0
    """
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    return dl


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ChaosXORModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dl = build_xor_dataset()

    for epoch in range(1, 501):
        loss = train_epoch(
            model,
            dl,
            optimizer,
            device=device,
            firing_reg_weight=0.1,
            firing_target_rate=0.2,
        )

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | loss = {loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        ).to(device)
        logits, spikes = model(x_test)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

    print("\nXOR results:")
    print("inputs:")
    print(x_test.cpu())
    print("probabilities:")
    print(probs.cpu())
    print("predictions:", preds.cpu().tolist())


if __name__ == "__main__":
    main()
