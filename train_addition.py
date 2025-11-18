import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

TOKENS = list("0123456789+?=")
VOCAB = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
for token in TOKENS:
    VOCAB[token] = len(VOCAB)

MAX_SEQ_LEN = 5  # e.g., "2+3=?"
NUM_CLASSES = 19  # sums from 0 to 18 inclusive

# Training configuration
NUM_EPOCHS = 100
FAIL_PROB = 0.999  # Probability of neuron failure during training


def expression_to_tensor(expression: str) -> torch.Tensor:
    indices = [VOCAB.get(ch, UNK_IDX) for ch in expression]
    if len(indices) < MAX_SEQ_LEN:
        indices += [PAD_IDX] * (MAX_SEQ_LEN - len(indices))
    return torch.tensor(indices[:MAX_SEQ_LEN], dtype=torch.long)


def generate_samples() -> List[Tuple[int, str]]:
    samples: List[Tuple[int, str]] = []
    for a in range(10):
        for b in range(10):
            if a == 2 and b == 2:
                continue
            label = a + b
            expr_primary = f"{a}+{b}=?"
            expr_swapped = f"{b}+{a}=?"
            samples.append((label, expr_primary))
            samples.append((label, expr_swapped))
    random.shuffle(samples)
    return samples


class AdditionDataset(Dataset):
    def __init__(self, samples: List[Tuple[int, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label, expr = self.samples[idx]
        return expression_to_tensor(expr), label


def split_samples(samples: List[Tuple[int, str]], val_ratio=0.1, test_ratio=0.1):
    total = len(samples)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size
    train = samples[:train_size]
    val = samples[train_size : train_size + val_size]
    test = samples[train_size + val_size :]
    return train, val, test


class ChaosAdditionModel(nn.Module):
    def __init__(self, embed_dim=64, hidden=512, ticks=10, fail_prob=0.0, symbolic_mode=False):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), embed_dim, padding_idx=PAD_IDX)
        
        # Force fail_prob=0 in symbolic mode
        effective_fail_prob = 0.0 if symbolic_mode else fail_prob
        
        neuron_params = ChaosNeuronParams(
            threshold=0.5,
            noise_std=0.1,
            fail_prob=effective_fail_prob,
            decay=0.01,
            refractory_decay=0.98,
        )
        cortex_params = CortexParams(
            input_size=embed_dim,
            hidden_sizes=[hidden, hidden//2],
            neuron=neuron_params,
        )
        self.cortex = ChaosCortex(cortex_params)
        self.readout = nn.Sequential(
            nn.Linear(hidden//2, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, NUM_CLASSES)
        )
        # Initialize weights
        for layer in self.readout:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu' if layer.out_features != NUM_CLASSES else 'linear')
                nn.init.constant_(layer.bias, 0.0)
        self.ticks = ticks
        self.symbolic_mode = symbolic_mode

    def forward(self, tokens):
        emb = self.embedding(tokens)
        mask = (tokens != PAD_IDX).unsqueeze(-1)
        valid = mask.sum(dim=1).clamp(min=1)
        avg_emb = (emb * mask).sum(dim=1) / valid

        batch = tokens.size(0)
        ticks = self.ticks
        expanded = avg_emb.unsqueeze(1).expand(batch, ticks, -1).reshape(batch * ticks, -1)

        out, _, layer_spikes = self.cortex(expanded, None)
        spikes = layer_spikes[-1] if layer_spikes else out
        spikes = spikes.view(batch, ticks, -1)

        avg_spikes = spikes.mean(dim=1)
        logits = self.readout(avg_spikes)
        return logits


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total = correct = running_loss = 0
    optimizer.zero_grad()
    
    # Gradient clipping value
    max_grad_norm = 1.0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass with gradient accumulation
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        # Calculate metrics
        with torch.no_grad():
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
            running_loss += loss.item()
            
            # Optional: Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | '
                      f'Acc: {100. * pred.eq(targets).sum().item() / targets.size(0):.2f}%')

    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    model.eval()
    total = correct = running_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return running_loss / len(dataloader), correct / total


def setup_experiment(base="experiments/addition"):
    exp_dir = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return exp_dir


def evaluate_secret(model, device):
    expression = "2+2=?"
    tokens = expression_to_tensor(expression).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tokens)
        pred = logits.argmax(dim=1).item()
    print(f'Secret expression "{expression}" predicted sum: {pred}')


def main():
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = generate_samples()
    train_samples, val_samples, test_samples = split_samples(samples)

    train_ds = AdditionDataset(train_samples)
    val_ds = AdditionDataset(val_samples)
    test_ds = AdditionDataset(test_samples)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128)
    test_dl = DataLoader(test_ds, batch_size=128)

    # Initialize model with larger capacity
    # In main():
    model = ChaosAdditionModel(
        embed_dim=128,
        hidden=2048,
        ticks=20,
        fail_prob=FAIL_PROB,
        symbolic_mode=True  # Set to False for pattern-based tasks
    ).to(device)
    
    # Optimizer with adjusted learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,               # Slightly lower learning rate
        weight_decay=1e-4,     # L2 regularization
        eps=1e-8               # For numerical stability
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,           # Maximum learning rate
        steps_per_epoch=len(train_dl),
        epochs=NUM_EPOCHS,
        pct_start=0.1,         # Warmup for 10% of training
        anneal_strategy='cos',  # Cosine annealing
        final_div_factor=10.0,  # Final LR = max_lr/10
        div_factor=10.0,        # Initial LR = max_lr/10
        three_phase=False)

    best_val = 0.0
    exp_dir = setup_experiment()
    print(f"Experiment directory: {exp_dir}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, device)
        val_loss, val_acc = evaluate(model, val_dl, device)
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), exp_dir / "checkpoints" / "best.pt")

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

    test_loss, test_acc = evaluate(model, test_dl, device)
    print(f"Final Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | Best Val: {best_val*100:.2f}%")
    evaluate_secret(model, device)


if __name__ == "__main__":
    main()
