import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import sys
import pickle
import json
from datetime import datetime
from pathlib import Path

# Make sure we can import chaosnet
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex

# ------------------------------------------------------------------------------
# Multi-tick ChaosNet MNIST model
# ------------------------------------------------------------------------------

class ChaosMNISTModel(nn.Module):
    def __init__(self, ticks=5, hidden=256):  
        super().__init__()
        self.ticks = ticks

        # Optimized neuron parameters
        neuron_params = ChaosNeuronParams(
            threshold=0.5,
            noise_std=0.01,
            fail_prob=0.999,
            decay=0.02,
            refractory_decay=0.95,
        )

        # Single hidden layer instead of two
        cortex_params = CortexParams(
            input_size=28 * 28,
            hidden_sizes=[hidden],  
            neuron=neuron_params,
        )

        self.cortex = ChaosCortex(cortex_params)
        
        # Initialize readout
        self.readout = nn.Linear(hidden, 10)
        nn.init.kaiming_normal_(self.readout.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.readout.bias, 0.0)

    def forward(self, x, collect_spikes=False):
        """
        x: (batch, 1, 28, 28) images
        collect_spikes: if True, returns spikes for visualization
        """
        batch = x.size(0)
        x = x.view(batch, -1)

        ticks = self.ticks
        expanded = x.unsqueeze(1).expand(batch, ticks, -1).reshape(batch * ticks, -1)

        out, _, layer_spikes = self.cortex(expanded, None)
        spikes = layer_spikes[0] if layer_spikes else out
        spikes = spikes.view(batch, ticks, -1)

        avg_spikes = spikes.mean(dim=1)
        logits = self.readout(avg_spikes)

        if collect_spikes:
            spike_stack = spikes.permute(1, 0, 2).detach()
            return logits, spike_stack
        # Maintain previous shape contract (ticks, batch, hidden)
        return logits, avg_spikes.unsqueeze(0)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, device, accumulation_steps=4):
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs, _ = model(inputs)
        loss = F.cross_entropy(outputs, targets) / accumulation_steps
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistics (multiply by accumulation_steps to get correct scale)
        with torch.no_grad():
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item() * accumulation_steps
    
    # Final metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def test(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def setup_experiment():
    """Set up experiment directories and configuration."""
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/exp_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoints and visualizations directories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    return exp_dir


def get_dataloaders(batch_size=32, val_split=0.1):
    """Get MNIST dataloaders with optional validation split."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, transform=transform)
    
    # Split training set into train/val
    if val_split > 0:
        val_size = int(len(train_ds) * val_split)
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_size, val_size])
    else:
        val_ds = None
    
    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=2) if val_ds else None
    
    return train_dl, val_dl, test_dl


def save_training_log(train_losses, train_accs, val_losses, val_accs, path):
    log_data = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }
    with open(path, 'wb') as f:
        pickle.dump(log_data, f)


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = setup_experiment()
    print(f"Experiment directory: {exp_dir}")
    
    # Get dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=64, val_split=0.1)
    
    # Initialize model with more stable parameters
    model = ChaosMNISTModel(ticks=5, hidden=512).to(device)
    
    # Use AdamW with weight decay and lower initial learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,  # Reduced from 1e-3
        weight_decay=1e-4
    )
    
    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_dl),
        epochs=15,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    num_epochs = 15
    
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_acc = 0.0
        if val_dl:
            val_acc = test(model, val_dl, device)
            val_accs.append(val_acc)
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), exp_dir / "checkpoints" / "best_model.pt")
        
        # Test set evaluation (less frequent)
        if epoch % 5 == 0 or epoch == num_epochs:
            test_acc = test(model, test_dl, device)
            print(f"Test Acc: {test_acc*100:.2f}%")
        
        # Print progress
        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}%")
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc,
            }
            torch.save(checkpoint, exp_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt")
            
            # Save training logs
            save_training_log(
                train_losses, train_accs, 
                val_losses if val_dl else None,
                val_accs if val_dl else None,
                path=exp_dir / "training_log.pkl"
            )
    
    # Final save
    torch.save(model.state_dict(), exp_dir / "final_model.pt")
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Final model and logs saved to: {exp_dir}")


if __name__ == "__main__":
    main()
